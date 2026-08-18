"""Exercise the rule-based chat-register rewriter on tiny strings, not real corpora."""

# Import importlib to load the CLI module from scripts/ without adding it to pythonpath.
import importlib.util

# Import json to parse urls_json cells written by the rewriter CLI.
import json

# Import Path for fixture directories and the CLI module location.
from pathlib import Path

# Import pandas to build a tiny processed-corpus CSV for the CLI integration test.
import pandas as pd

# Import pytest for exception assertions.
import pytest

from secure_chat_ml.chat_register import REWRITE_METHOD, rewrite_message

# Phrases a ham lunch email must not gain from the rewriter.
_PHISHING_INSERTIONS = (
    "verify your",
    "password now",
    "urgent",
    "prize",
    "suspended",
    "gift card",
    "wire money",
    "act now",
    "claim your",
)


# Confirm a phishing email with a login URL becomes a short DM that still has that URL.
def test_phishing_email_rewrite_keeps_the_original_url() -> None:
    """Assert https://evil.example/login survives a rule_based_v1 rewrite."""

    # Fake email with headers, a greeting, a login URL, and an unsubscribe footer.
    email = (
        "From: attacker@example.com\n"
        "To: victim@example.com\n"
        "Subject: Account notice\n"
        "\n"
        "Dear Customer,\n"
        "Please login at https://evil.example/login to continue using your account.\n"
        "If you are not the intended recipient, delete this message.\n"
        "Click here to unsubscribe from this mailing list.\n"
        "Best regards,\n"
        "Support\n"
    )
    # Rewrite as a scam-labeled row so a casual opener is allowed.
    rewritten = rewrite_message(email, label=1)
    # The rewrite must produce a non-empty DM line.
    assert rewritten is not None
    # The original URL must still be present so link features stay honest.
    assert "https://evil.example/login" in rewritten
    # The rewrite must stay within the chat-scale cap.
    assert len(rewritten) <= 400
    # Email chrome should not dominate the DM line.
    assert "Dear Customer" not in rewritten
    assert "unsubscribe" not in rewritten.lower()
    assert "Best regards" not in rewritten


# Confirm a ham lunch email stays a lunch DM and gains no phishing verbs.
def test_ham_lunch_email_rewrite_adds_no_phishing_verbs() -> None:
    """Assert a legitimate lunch email does not gain scam boilerplate."""

    # Fake ham email about lunch, with a greeting and a closing.
    email = (
        "From: alex@example.com\n"
        "To: sam@example.com\n"
        "Subject: Lunch\n"
        "\n"
        "Hi Sam,\n"
        "Want to get lunch tomorrow at noon? I'm thinking the usual cafe.\n"
        "Best regards,\n"
        "Alex\n"
    )
    # Rewrite as a legitimate row so urgency/scam wrappers are forbidden.
    rewritten = rewrite_message(email, label=0)
    # The rewrite must produce a non-empty DM line.
    assert rewritten is not None
    # The lunch meaning must survive shortening.
    assert "lunch" in rewritten.lower()
    # The rewriter must not insert phishing verbs that were not in the source.
    rewritten_lower = rewritten.lower()
    for phrase in _PHISHING_INSERTIONS:
        # None of the forbidden insertions may appear in the ham rewrite.
        assert phrase not in rewritten_lower
    # Formal greetings should be stripped.
    assert "Dear" not in rewritten
    # The rewrite method identifier is documented for README/audit rows.
    assert REWRITE_METHOD == "rule_based_v1"


# Confirm empty input is skipped rather than emitting a blank training row.
def test_rewrite_message_returns_none_for_empty_text() -> None:
    """Assert blank source text is dropped."""

    # Empty and whitespace-only strings should both be skipped.
    assert rewrite_message("", label=0) is None
    assert rewrite_message("   ", label=1) is None
    assert rewrite_message(None, label=0) is None  # type: ignore[arg-type]


# Load scripts/rewrite_chat_register.py as a module for CLI-level tests.
def _load_rewrite_script():
    """Return the rewrite_chat_register CLI module loaded from scripts/."""

    # Resolve the CLI path relative to this test file.
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "rewrite_chat_register.py"
    # Build a module spec from the file location.
    spec = importlib.util.spec_from_file_location("rewrite_chat_register", script_path)
    # Fail clearly if the CLI file is missing.
    assert spec is not None and spec.loader is not None
    # Create an empty module object for the loader to populate.
    module = importlib.util.module_from_spec(spec)
    # Execute the CLI module (imports secure_chat_ml, does not rewrite yet).
    spec.loader.exec_module(module)
    # Return the loaded module so tests can call rewrite_corpora.
    return module


# Confirm the CLI refuses to read or write the locked chat_eval directory.
def test_rewrite_corpora_refuses_chat_eval_paths(tmp_path: Path) -> None:
    """Assert data/chat_eval cannot be a rewrite source or destination."""

    # Load the CLI module once for this test.
    module = _load_rewrite_script()
    # A path whose parts include chat_eval must be rejected.
    with pytest.raises(ValueError, match="chat_eval"):
        module.rewrite_corpora(tmp_path / "chat_eval", tmp_path / "processed_chat")
    with pytest.raises(ValueError, match="chat_eval"):
        module.rewrite_corpora(tmp_path / "processed", tmp_path / "data" / "chat_eval")


# Confirm the CLI rewrites a tiny processed CSV and keeps labels plus URLs.
def test_rewrite_corpora_writes_chat_register_csv(tmp_path: Path) -> None:
    """Assert a synthetic processed dir becomes processed_chat with labels intact."""

    # Load the CLI module.
    module = _load_rewrite_script()
    # Create a tiny processed corpus shaped like the real schema.
    input_dir = tmp_path / "processed"
    input_dir.mkdir()
    # One ham lunch email and one phishing email with a URL.
    frame = pd.DataFrame(
        {
            "message_id": ["syn-0", "syn-1"],
            "text": [
                "Subject: Lunch\n\nWant to get lunch tomorrow at noon?",
                "Dear Customer, login at https://evil.example/login now.",
            ],
            "label": [0, 1],
            "original_label": ["ham", "spam"],
            "source": ["synthetic_test_corpus", "synthetic_test_corpus"],
            "split": ["unassigned", "unassigned"],
        }
    )
    # Write the synthetic processed CSV.
    frame.to_csv(input_dir / "synthetic.csv", index=False)
    # Point the rewriter at the temporary directories.
    output_dir = tmp_path / "processed_chat"
    # Run the rewrite; it must not touch chat_eval.
    log = module.rewrite_corpora(input_dir, output_dir)
    # Two source rows should both survive as non-empty rewrites.
    assert log["rows_in"] == 2
    assert log["rows_out"] == 2
    assert log["chat_eval_touched"] is False
    # The rewrite method must be the documented rule_based_v1 identifier.
    assert log["rewrite_method"] == "rule_based_v1"
    # Read the written CSV back.
    written = pd.read_csv(output_dir / "synthetic_test_corpus.csv")
    # Schema columns plus rewrite metadata must be present.
    for column in (
        "message_id",
        "text",
        "label",
        "original_label",
        "source",
        "split",
        "source_message_id",
        "rewrite_method",
        "urls_json",
    ):
        # Every expected column must exist in the written file.
        assert column in written.columns
    # Labels must be copied, never flipped.
    assert set(written["label"].tolist()) == {0, 1}
    # The phishing URL must still be in the scam row's rewritten text.
    scam_row = written.loc[written["label"] == 1].iloc[0]
    assert "https://evil.example/login" in scam_row["text"]
    # urls_json on the scam row must record that URL.
    assert "https://evil.example/login" in json.loads(scam_row["urls_json"])
    # The ham rewrite must still mention lunch and must not gain phishing verbs.
    ham_row = written.loc[written["label"] == 0].iloc[0]
    assert "lunch" in ham_row["text"].lower()
    for phrase in _PHISHING_INSERTIONS:
        assert phrase not in ham_row["text"].lower()
