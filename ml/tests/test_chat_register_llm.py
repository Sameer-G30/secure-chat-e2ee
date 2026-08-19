"""Exercise the LLM chat-register rewriter with a fake generate callback.

These tests never start Ollama, never download models, and never read
`data/chat_eval/chat_style_eval_v1.csv`. A tiny fixture rewrite documents
the expected schema.
"""

# Import importlib to load the CLI module from scripts/ without adding it to pythonpath.
import importlib.util

# Import json to parse urls_json cells written by the rewriter CLI.
import json

# Import urllib.request so tests can prove the LLM path does not fetch URLs.
import urllib.request

# Import Path for fixture directories and the CLI module location.
from pathlib import Path

# Import pandas to build a tiny processed-corpus CSV for the CLI integration test.
import pandas as pd

# Import pytest for exception assertions and monkeypatching.
import pytest

from secure_chat_ml.chat_register import MAX_REWRITE_CHARS
from secure_chat_ml.chat_register_llm import (
    _OLLAMA_OPTIONS,
    FALLBACK_REWRITE_METHOD,
    LLM_MAX_REWRITE_CHARS,
    OLLAMA_NUM_PREDICT,
    RESEARCH_RETRY_PREAMBLE,
    REWRITE_METHOD,
    SYSTEM_PROMPT,
    assert_local_ollama_host,
    assert_not_chat_eval_path,
    clean_llm_output,
    rewrite_message_llm,
)

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

# Tiny fixture rewrite kept on disk so tests do not need Ollama or downloads.
_FIXTURE_REWRITE = Path(__file__).resolve().parent / "fixtures" / "llm_intent_v1_tiny.csv"

# Fake phishing email used by several tests (must contain this exact URL).
_PHISHING_EMAIL = (
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

# Fake ham lunch email used by several tests.
_LUNCH_EMAIL = (
    "From: alex@example.com\n"
    "To: sam@example.com\n"
    "Subject: Lunch\n"
    "\n"
    "Hi Sam,\n"
    "Want to get lunch tomorrow at noon? I'm thinking the usual cafe.\n"
    "Best regards,\n"
    "Alex\n"
)


# Intent-preserving fake LLM: paraphrase, keep the URL, do not call Ollama.
def _fake_generate(_system: str, user: str) -> str:
    """Return a short DM derived from the user prompt without network I/O."""

    # Phishing source: write a credential-theft DM that still contains the URL.
    if "https://evil.example/login" in user:
        # Keep the URL character-for-character as the post-condition also requires.
        return "hey can you open this https://evil.example/login i think my login expired"
    # Lunch source: write a lunch DM with no phishing verbs.
    if "lunch" in user.lower():
        # Preserve the lunch request in chat register.
        return "want to get lunch tomorrow at noon? thinking the usual cafe"
    # Fallback for other synthetic rows in the CLI test.
    return "ok sounds good, see you then"


# Load scripts/rewrite_chat_register_llm.py as a module for CLI-level tests.
def _load_rewrite_script():
    """Return the rewrite_chat_register_llm CLI module loaded from scripts/."""

    # Resolve the CLI path relative to this test file.
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "rewrite_chat_register_llm.py"
    )
    # Build a module spec from the file location.
    spec = importlib.util.spec_from_file_location("rewrite_chat_register_llm", script_path)
    # Fail clearly if the CLI file is missing.
    assert spec is not None and spec.loader is not None
    # Create an empty module object for the loader to populate.
    module = importlib.util.module_from_spec(spec)
    # Execute the CLI module (imports secure_chat_ml, does not rewrite yet).
    spec.loader.exec_module(module)
    # Return the loaded module so tests can call rewrite_corpora.
    return module


# Confirm the documented rewrite identifier is llm_intent_v1, not rule_based_v1.
def test_rewrite_method_identifier_is_llm_intent_v1() -> None:
    """Assert successful LLM rewrites stamp llm_intent_v1."""

    # The identifier must match the training-report rewrite_method field.
    assert REWRITE_METHOD == "llm_intent_v1"
    # Refusal fallbacks must use a distinct stamp, not overwrite rule_based_v1.
    assert FALLBACK_REWRITE_METHOD == "rule_based_v1_fallback"


# Confirm generation length constants: 160 truncated DMs; the LLM cap is not 400.
def test_ollama_num_predict_is_400_and_llm_cap_is_600() -> None:
    """Assert num_predict=400 and the LLM-only cap is 600, not rule_based 400."""

    # The public constant must be 400 so ~600-char DMs plus URLs are not truncated.
    assert OLLAMA_NUM_PREDICT == 400
    # The options dict sent to Ollama must use that constant.
    assert _OLLAMA_OPTIONS["num_predict"] == 400
    # The LLM prose cap must be 600 characters (URLs may make the stored DM longer).
    assert LLM_MAX_REWRITE_CHARS == 600
    # The rule-based rewriter must keep its own 400-character cap.
    assert MAX_REWRITE_CHARS == 400
    # The LLM path must not reuse the rule-based cap as its default.
    assert LLM_MAX_REWRITE_CHARS != MAX_REWRITE_CHARS


# Confirm the system prompt tells the model to write a DM, not a summary.
def test_system_prompt_asks_for_a_new_chat_message() -> None:
    """Assert the generation spec forbids email salvage and requires a new DM."""

    # The model must write the message itself, not describe the email.
    assert "Write the message itself" in SYSTEM_PROMPT
    # The model must not keep RFC822 chrome.
    assert "From/To/Subject" in SYSTEM_PROMPT
    # Legitimate rows must not gain phishing verbs the source did not have.
    assert "label=0" in SYSTEM_PROMPT.lower() or "If label=0" in SYSTEM_PROMPT
    # Length target is 40–600 characters of chat prose (URLs extra).
    assert "40-600" in SYSTEM_PROMPT
    # Scam rows must paraphrase the existing labeled message, not invent a new scam.
    assert "EXISTING labeled message" in SYSTEM_PROMPT
    # Research framing must make clear this is not scam creation.
    assert "You are not creating a scam" in SYSTEM_PROMPT
    # The research retry preamble must be register-only, not "write a new scam".
    assert "university scam-detection dataset" in RESEARCH_RETRY_PREAMBLE
    assert "Only rewrite STYLE" in RESEARCH_RETRY_PREAMBLE


# Confirm a phishing email with a login URL becomes a short DM that still has that URL.
def test_phishing_email_llm_rewrite_keeps_the_original_url() -> None:
    """Assert https://evil.example/login survives an llm_intent_v1 fake rewrite."""

    # Rewrite as a scam-labeled row so a scam-register DM is allowed.
    result = rewrite_message_llm(_PHISHING_EMAIL, label=1, generate=_fake_generate)
    # The rewrite must succeed rather than being counted as llm_failed.
    assert result.status == "ok"
    # The rewrite must produce a non-empty DM line.
    assert result.text is not None
    # The original URL must still be present so link features stay honest.
    assert "https://evil.example/login" in result.text
    # A successful model DM must stamp llm_intent_v1, not the fallback method.
    assert result.rewrite_method == REWRITE_METHOD
    # Label is not returned; callers copy it unchanged (this test used 1).
    assert result.status == "ok"
    # Email chrome should not dominate the fake DM.
    assert "Dear Customer" not in result.text
    assert "Best regards" not in result.text


# Confirm a ham lunch email stays a lunch DM and gains no phishing verbs.
def test_ham_lunch_email_llm_rewrite_adds_no_phishing_verbs() -> None:
    """Assert a legitimate lunch email does not gain scam boilerplate."""

    # Rewrite as a legitimate row so urgency/scam wrappers are forbidden.
    result = rewrite_message_llm(_LUNCH_EMAIL, label=0, generate=_fake_generate)
    # The rewrite must succeed.
    assert result.status == "ok"
    # The rewrite must produce a non-empty DM line.
    assert result.text is not None
    # The lunch meaning must survive the paraphrase.
    assert "lunch" in result.text.lower()
    # The rewriter must not insert phishing verbs that were not in the source.
    rewritten_lower = result.text.lower()
    # Walk the frozen insertion list used by the rule-based tests as well.
    for phrase in _PHISHING_INSERTIONS:
        # None of the forbidden insertions may appear in the ham rewrite.
        assert phrase not in rewritten_lower


# Confirm ham post-conditions strip phishing verbs a sloppy model inserted.
def test_ham_rewrite_strips_inserted_phishing_verbs() -> None:
    """Assert label=0 drops 'verify your' when the source lunch email lacked it."""

    # Fake model that ignores the ham constraint (post-conditions must repair it).
    def _sloppy(_system: str, _user: str) -> str:
        # Insert a phishing frame that was not in the lunch source.
        return "want lunch tomorrow, verify your password now at the cafe"

    # Rewrite as legitimate so the ham-safety strip runs.
    result = rewrite_message_llm(_LUNCH_EMAIL, label=0, generate=_sloppy)
    # The row must still be kept after the strip.
    assert result.status == "ok"
    assert result.text is not None
    # Lunch meaning must remain.
    assert "lunch" in result.text.lower()
    # Inserted phishing frames must be gone.
    assert "verify your" not in result.text.lower()
    assert "password now" not in result.text.lower()


# Confirm missing URL in model output is re-attached by the post-condition.
def test_missing_url_in_model_output_is_reattached() -> None:
    """Assert a dropped https://evil.example/login is appended, not dropped."""

    # Fake model that paraphrases the scam but forgets the URL.
    def _forget_url(_system: str, _user: str) -> str:
        # Return a credential-theft DM without the URL substring.
        return "hey can you login so your account stays open"

    # Rewrite the phishing email; post-conditions must append the URL.
    result = rewrite_message_llm(_PHISHING_EMAIL, label=1, generate=_forget_url)
    # The row must be kept, not skipped.
    assert result.status == "ok"
    assert result.text is not None
    # The original URL must appear character-for-character after post-conditions.
    assert "https://evil.example/login" in result.text
    # The append flag must record that Python re-attached the URL.
    assert result.urls_appended is True
    # The row is kept as a successful LLM rewrite (label stays 1 at the caller).
    assert result.rewrite_method == REWRITE_METHOD


# Confirm wrapping quotes from the model are stripped before the DM is stored.
def test_clean_llm_output_strips_wrapping_quotes() -> None:
    """Assert a quoted model response becomes a bare DM line."""

    # Models often wrap the DM in ASCII double quotes despite the spec.
    cleaned = clean_llm_output('"want to get lunch tomorrow at noon?"')
    # The wrapping quotes must be gone.
    assert cleaned == "want to get lunch tomorrow at noon?"


# Confirm empty/garbage output is retried once, then kept via rule_based fallback.
def test_garbage_output_is_retried_then_falls_back() -> None:
    """Assert two unusable model responses still keep the row via rule_based_v1."""

    # Count calls so we know the research retry happened.
    calls = {"n": 0}

    # Always return a meta-summary, which the spec forbids.
    def _garbage(_system: str, _user: str) -> str:
        # Increment the call counter.
        calls["n"] += 1
        # Return a summary rather than a DM.
        return "this email is about lunch"

    # Rewrite the lunch email; both LLM attempts should be unusable.
    result = rewrite_message_llm(_LUNCH_EMAIL, label=0, generate=_garbage)
    # Two attempts is the documented retry-once policy (initial + research retry).
    assert calls["n"] == 2
    # The row must be kept via the deterministic fallback, not dropped.
    assert result.status == "ok"
    assert result.text is not None
    # The stored text must not be the garbage summary.
    assert "this email is about" not in result.text.lower()
    # Lunch meaning must survive the rule_based_v1 fallback.
    assert "lunch" in result.text.lower()
    # The stamp must record that the LLM did not produce the DM.
    assert result.rewrite_method == FALLBACK_REWRITE_METHOD


# Confirm llama-style safety refusals are never stored; the row is kept another way.
def test_assistant_refusal_is_not_stored_as_training_text() -> None:
    """Assert a double refusal falls back instead of storing the refusal string."""

    # A callback that always refuses instead of writing the requested scam DM.
    def _refuse(_system: str, _user: str) -> str:
        # Return a typical Llama 3.2 safety refusal.
        return (
            "I cannot write a message that could be used in a scam. "
            "Is there anything else I can help you with?"
        )

    # Rewrite a phishing email; both LLM attempts should be rejected as garbage.
    result = rewrite_message_llm(_PHISHING_EMAIL, label=1, generate=_refuse)
    # The refusal must not become a training row labeled scam.
    assert result.text is not None
    assert "cannot write a message that could be used in a scam" not in result.text.lower()
    # The labeled scam row must still be kept.
    assert result.status == "ok"
    assert result.rewrite_method == FALLBACK_REWRITE_METHOD
    # Original URLs must still appear after the fallback.
    assert "https://evil.example/login" in result.text


# Confirm a first garbage attempt can recover on retry.
def test_garbage_then_good_output_is_accepted_on_retry() -> None:
    """Assert the second generate call is used when the first is unusable."""

    # Count calls so the test can see the retry.
    calls = {"n": 0}

    # Fail once, then return a valid lunch DM.
    def _flaky(_system: str, user: str) -> str:
        # Increment the call counter.
        calls["n"] += 1
        # First attempt is a forbidden summary.
        if calls["n"] == 1:
            # Return garbage so rewrite_message_llm retries.
            return ""
        # Second attempt is a usable lunch DM.
        return "want to get lunch tomorrow at noon? thinking the usual cafe"

    # Rewrite the lunch email with the flaky callback.
    result = rewrite_message_llm(_LUNCH_EMAIL, label=0, generate=_flaky)
    # Both attempts must have run.
    assert calls["n"] == 2
    # The recovered DM must be stored.
    assert result.status == "ok"
    assert result.text is not None
    assert "lunch" in result.text.lower()
    # Recovery on the research retry still counts as an LLM DM, not fallback.
    assert result.rewrite_method == REWRITE_METHOD


# Confirm a first-attempt scam refusal can recover on the research retry.
def test_refusal_then_research_retry_keeps_scam_row() -> None:
    """Assert a refused first call plus a usable retry keeps label=1 without the refusal."""

    # Record prompts so the test can see the research retry framing.
    calls = {"n": 0, "users": []}

    # Refuse once, then return a credential-theft DM that still contains the URL.
    def _once_refuse(_system: str, user: str) -> str:
        # Increment the call counter.
        calls["n"] += 1
        # Remember the user prompt for the research-retry assertion.
        calls["users"].append(user)
        # First attempt is a typical Llama safety refusal.
        if calls["n"] == 1:
            # Return unusable text so rewrite_message_llm retries with research framing.
            return "I cannot write a message that could be used in a scam."
        # Second attempt is a usable scam DM that keeps the original URL.
        return "hey can you open this https://evil.example/login i think my login expired"

    # Rewrite the phishing email as a labeled scam row.
    result = rewrite_message_llm(_PHISHING_EMAIL, label=1, generate=_once_refuse)
    # Both generate calls must have run (standard prompt, then research retry).
    assert calls["n"] == 2
    # The research retry prompt must not ask the model to write a new scam.
    assert "university scam-detection dataset" in calls["users"][1]
    assert "Only rewrite STYLE" in calls["users"][1]
    # The row must be kept as a real DM, not the refusal string.
    assert result.status == "ok"
    assert result.text is not None
    assert "cannot write a message that could be used in a scam" not in result.text.lower()
    # The original URL must still be present.
    assert "https://evil.example/login" in result.text
    # A successful research retry is still an LLM rewrite.
    assert result.rewrite_method == REWRITE_METHOD


# Confirm a double refusal on a URL-bearing scam uses rule_based fallback and keeps the URL.
def test_double_refusal_uses_rule_based_fallback_and_keeps_urls() -> None:
    """Assert two refusals fall back, keep the row, and preserve https://evil.example/login."""

    # Always refuse so both the first prompt and the research retry are unusable.
    def _refuse(_system: str, _user: str) -> str:
        # Return the same safety refusal on every generate call.
        return "I cannot write a message that could be used in a scam."

    # Rewrite the phishing email; fallback must keep the labeled scam row.
    result = rewrite_message_llm(_PHISHING_EMAIL, label=1, generate=_refuse)
    # The row must be kept rather than dropped as llm_failed.
    assert result.status == "ok"
    assert result.text is not None
    # The stored text must not be the refusal.
    assert "cannot write" not in result.text.lower()
    # The original URL must still appear after the fallback.
    assert "https://evil.example/login" in result.text
    # The stamp must record that rule_based_v1 produced this row.
    assert result.rewrite_method == FALLBACK_REWRITE_METHOD


# Confirm empty source text is skipped without calling the generator.
def test_rewrite_message_llm_returns_empty_for_blank_text() -> None:
    """Assert blank source text is dropped before any generate call."""

    # A generator that fails the test if it is invoked.
    def _must_not_run(_system: str, _user: str) -> str:
        # Calling the LLM on empty source would be wasted work.
        raise AssertionError("generate must not be called for empty source text")

    # Empty, whitespace-only, and None must all be skipped.
    assert rewrite_message_llm("", label=0, generate=_must_not_run).status == "empty"
    assert rewrite_message_llm("   ", label=1, generate=_must_not_run).status == "empty"
    assert rewrite_message_llm(None, label=0, generate=_must_not_run).status == "empty"  # type: ignore[arg-type]


# Confirm cloud LLM hosts are refused before any HTTP call.
def test_non_local_ollama_host_is_refused() -> None:
    """Assert OpenAI-style hosts cannot receive corpus text."""

    # A cloud API origin must be rejected.
    with pytest.raises(ValueError, match="non-local"):
        # Use a well-known cloud host that must never see this corpus.
        assert_local_ollama_host("https://api.openai.com")


# Confirm chat_eval path parts are refused at the library layer.
def test_assert_not_chat_eval_path_refuses_locked_eval_dir(tmp_path: Path) -> None:
    """Assert data/chat_eval cannot be a rewrite source or destination."""

    # A nested chat_eval directory must be rejected.
    with pytest.raises(ValueError, match="chat_eval"):
        # Point at a path whose parts include chat_eval.
        assert_not_chat_eval_path(tmp_path / "data" / "chat_eval")


# Confirm rewrite_message_llm with a fake callback never opens a socket.
def test_injected_generate_does_not_call_urlopen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert the fake-LLM path does not HTTP-fetch URLs or call Ollama."""

    # Fail immediately if any code path calls urlopen.
    def _forbid_urlopen(*_args: object, **_kwargs: object) -> None:
        # Raise so a regression that starts fetching cannot pass silently.
        raise AssertionError("urlopen must not be called when generate is injected")

    # Patch the stdlib fetch entry point for the duration of this test.
    monkeypatch.setattr(urllib.request, "urlopen", _forbid_urlopen)
    # Rewrite both a ham and a phishing example through the fake callback.
    ham = rewrite_message_llm(_LUNCH_EMAIL, label=0, generate=_fake_generate)
    # The ham rewrite must succeed without network I/O.
    assert ham.status == "ok"
    # Rewrite the phishing example as well.
    scam = rewrite_message_llm(_PHISHING_EMAIL, label=1, generate=_fake_generate)
    # The scam rewrite must still contain the original URL.
    assert scam.text is not None and "https://evil.example/login" in scam.text


# Confirm the CLI refuses to read or write the locked chat_eval directory.
def test_rewrite_corpora_refuses_chat_eval_paths(tmp_path: Path) -> None:
    """Assert data/chat_eval cannot be a rewrite source or destination."""

    # Load the CLI module once for this test.
    module = _load_rewrite_script()
    # A path whose parts include chat_eval must be rejected.
    with pytest.raises(ValueError, match="chat_eval"):
        # Pass a fake generate so a failure cannot be an Ollama connection error.
        module.rewrite_corpora(
            tmp_path / "chat_eval",
            tmp_path / "processed_chat_llm",
            generate=_fake_generate,
            resume=False,
        )
    with pytest.raises(ValueError, match="chat_eval"):
        # Refuse writing into chat_eval as well.
        module.rewrite_corpora(
            tmp_path / "processed",
            tmp_path / "data" / "chat_eval",
            generate=_fake_generate,
            resume=False,
        )


# Confirm the CLI rewrites a tiny processed CSV and keeps labels plus URLs.
def test_rewrite_corpora_writes_llm_chat_register_csv(tmp_path: Path) -> None:
    """Assert a synthetic processed dir becomes processed_chat_llm with labels intact."""

    # Load the CLI module.
    module = _load_rewrite_script()
    # Create a tiny processed corpus shaped like the real schema.
    input_dir = tmp_path / "processed"
    # Create the input directory.
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
    output_dir = tmp_path / "processed_chat_llm"
    # Run the rewrite with the fake LLM; it must not touch chat_eval or Ollama.
    log = module.rewrite_corpora(
        input_dir,
        output_dir,
        generate=_fake_generate,
        resume=False,
        progress_every=1,
    )
    # Two source rows should both survive as non-empty rewrites.
    assert log["rows_in"] == 2
    assert log["rows_out"] == 2
    assert log["chat_eval_touched"] is False
    # Both rows were usable LLM DMs; nothing fell back or failed.
    assert log["llm_ok"] == 2
    assert log["llm_refused_then_fallback"] == 0
    assert log["llm_failed"] == 0
    assert log["dropped_empty"] == 0
    # Generation-length settings must be recorded on the log.
    assert log["num_predict"] == 400
    assert log["llm_max_rewrite_chars"] == 600
    # The rewrite method must be the documented llm_intent_v1 identifier.
    assert log["rewrite_method"] == "llm_intent_v1"
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
    # Every written row must stamp llm_intent_v1.
    assert set(written["rewrite_method"].tolist()) == {"llm_intent_v1"}


# Confirm --resume skips rows already marked ok without calling generate again.
def test_rewrite_corpora_resume_skips_completed_rows(tmp_path: Path) -> None:
    """Assert a second run with resume=True does not re-prompt finished ids."""

    # Load the CLI module.
    module = _load_rewrite_script()
    # Create a tiny processed corpus with one lunch row.
    input_dir = tmp_path / "processed"
    # Create the input directory.
    input_dir.mkdir()
    # A single ham row is enough to test resume.
    frame = pd.DataFrame(
        {
            "message_id": ["syn-0"],
            "text": ["Want to get lunch tomorrow at noon?"],
            "label": [0],
            "original_label": ["ham"],
            "source": ["synthetic_test_corpus"],
            "split": ["unassigned"],
        }
    )
    # Write the synthetic processed CSV.
    frame.to_csv(input_dir / "synthetic.csv", index=False)
    # Count generate calls across both runs.
    calls = {"n": 0}

    # Wrap the fake generator so we can count invocations.
    def _counted(system: str, user: str) -> str:
        # Increment the call counter.
        calls["n"] += 1
        # Delegate to the shared fake generate.
        return _fake_generate(system, user)

    # Point both runs at the same output directory so the checkpoint is reused.
    output_dir = tmp_path / "processed_chat_llm"
    # First run writes the checkpoint and the CSV.
    module.rewrite_corpora(
        input_dir,
        output_dir,
        generate=_counted,
        resume=False,
        progress_every=1,
    )
    # The first run must have called generate once.
    assert calls["n"] == 1
    # Second run with resume must skip the completed row.
    log = module.rewrite_corpora(
        input_dir,
        output_dir,
        generate=_counted,
        resume=True,
        progress_every=1,
    )
    # generate must not have been called again.
    assert calls["n"] == 1
    # The resume counter must record the skip.
    assert log["resumed_skipped"] == 1
    # The output CSV must still exist with the lunch DM.
    written = pd.read_csv(output_dir / "synthetic_test_corpus.csv")
    assert "lunch" in written.iloc[0]["text"].lower()


# Confirm the on-disk tiny fixture rewrite has the expected schema and labels.
def test_tiny_fixture_rewrite_has_schema_and_unflipped_labels() -> None:
    """Assert tests/fixtures/llm_intent_v1_tiny.csv is a usable schema sample."""

    # The fixture must exist so CI does not need a full rewrite.
    assert _FIXTURE_REWRITE.exists()
    # Load the tiny fixture rewrite.
    frame = pd.read_csv(_FIXTURE_REWRITE)
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
        # Every expected column must exist in the fixture.
        assert column in frame.columns
    # Labels must remain the original binary 0/1 pair.
    assert set(frame["label"].tolist()) == {0, 1}
    # The fixture must stamp llm_intent_v1.
    assert set(frame["rewrite_method"].tolist()) == {"llm_intent_v1"}
    # The scam row must keep the exact phishing URL.
    scam_row = frame.loc[frame["label"] == 1].iloc[0]
    assert "https://evil.example/login" in scam_row["text"]
    # The ham row must mention lunch and must not include phishing insertions.
    ham_row = frame.loc[frame["label"] == 0].iloc[0]
    assert "lunch" in ham_row["text"].lower()
    for phrase in _PHISHING_INSERTIONS:
        assert phrase not in ham_row["text"].lower()


# Confirm the LLM length cap keeps prose past the rule-based 400-character limit.
def test_llm_rewrite_cap_allows_prose_between_400_and_600_chars() -> None:
    """Assert a 500-character fake DM is kept, not sliced to the rule-based 400 cap."""

    # Build a lunch-themed DM longer than rule_based_v1 MAX_REWRITE_CHARS.
    long_dm = ("want to get lunch tomorrow at noon at the usual cafe? ") * 10
    # Sanity-check the fixture length sits between the two caps.
    assert 400 < len(long_dm) < 600

    # Fake model that returns the long DM unchanged.
    def _long(_system: str, _user: str) -> str:
        # Return chat prose that would be sliced if the LLM used the 400 cap.
        return long_dm

    # Rewrite the lunch email as legitimate so ham-safety still runs.
    result = rewrite_message_llm(_LUNCH_EMAIL, label=0, generate=_long)
    # The row must be kept.
    assert result.status == "ok"
    assert result.text is not None
    # The stored DM must remain longer than the rule-based 400-character cap.
    assert len(result.text) > MAX_REWRITE_CHARS
    # The stored DM must not exceed the LLM cap (no URLs on this ham row).
    assert len(result.text) <= LLM_MAX_REWRITE_CHARS


# Confirm a long model DM that omitted the URL still keeps the URL intact.
def test_long_rewrite_never_slices_original_urls() -> None:
    """Assert https://evil.example/login is appended in full when prose is over-long."""

    # Build prose long enough that a naive hard-slice would cut a trailing URL.
    long_prose = ("please login so your account stays open. ") * 40

    # Fake model that paraphrases the scam but forgets the URL.
    def _long_without_url(_system: str, _user: str) -> str:
        # Return over-long prose with no URL substring.
        return long_prose

    # Rewrite the phishing email; post-conditions must keep the URL verbatim.
    result = rewrite_message_llm(_PHISHING_EMAIL, label=1, generate=_long_without_url)
    # The row must be kept, not dropped.
    assert result.status == "ok"
    assert result.text is not None
    # The original URL must appear character-for-character (never sliced).
    assert "https://evil.example/login" in result.text
    # Python must have appended the omitted URL.
    assert result.urls_appended is True


# Confirm the CLI keeps a refused scam row via fallback and drops empty source.
def test_rewrite_corpora_fallback_keeps_scam_and_drops_empty(tmp_path: Path) -> None:
    """Assert refusals become rule_based_v1_fallback rows and empty source is dropped."""

    # Load the CLI module.
    module = _load_rewrite_script()
    # Create a tiny processed corpus with ham, scam, and empty source rows.
    input_dir = tmp_path / "processed"
    # Create the input directory.
    input_dir.mkdir()
    # Three rows: usable ham, refused scam with a URL, and empty source.
    frame = pd.DataFrame(
        {
            "message_id": ["syn-0", "syn-1", "syn-2"],
            "text": [
                "Subject: Lunch\n\nWant to get lunch tomorrow at noon?",
                "Dear Customer, login at https://evil.example/login now.",
                "   ",
            ],
            "label": [0, 1, 1],
            "original_label": ["ham", "spam", "spam"],
            "source": [
                "synthetic_test_corpus",
                "synthetic_test_corpus",
                "synthetic_test_corpus",
            ],
            "split": ["unassigned", "unassigned", "unassigned"],
        }
    )
    # Write the synthetic processed CSV.
    frame.to_csv(input_dir / "synthetic.csv", index=False)

    # Refuse scam rows; succeed on ham so the log has both llm_ok and fallback.
    def _refuse_scams(_system: str, user: str) -> str:
        # Scam-labeled prompts include the phishing URL in the user turn.
        if "https://evil.example/login" in user:
            # Return a refusal so the rewriter falls back for this row.
            return "I cannot write a message that could be used in a scam."
        # Ham lunch prompt: return a usable DM.
        return "want to get lunch tomorrow at noon? thinking the usual cafe"

    # Point the rewriter at the temporary directories.
    output_dir = tmp_path / "processed_chat_llm"
    # Run the rewrite with the mixed fake; it must not touch chat_eval or Ollama.
    log = module.rewrite_corpora(
        input_dir,
        output_dir,
        generate=_refuse_scams,
        resume=False,
        progress_every=1,
    )
    # Three source rows: one empty drop, two kept (LLM ham + fallback scam).
    assert log["rows_in"] == 3
    assert log["dropped_empty"] == 1
    assert log["llm_ok"] == 1
    assert log["llm_refused_then_fallback"] == 1
    assert log["llm_failed"] == 0
    assert log["rows_out"] == 2
    # Read the written CSV back.
    written = pd.read_csv(output_dir / "synthetic_test_corpus.csv")
    # Labels must still be the original 0/1 pair.
    assert set(written["label"].tolist()) == {0, 1}
    # The scam row must keep the URL and must not store the refusal string.
    scam_row = written.loc[written["label"] == 1].iloc[0]
    assert "https://evil.example/login" in scam_row["text"]
    assert "cannot write a message that could be used in a scam" not in scam_row["text"].lower()
    assert scam_row["rewrite_method"] == FALLBACK_REWRITE_METHOD
    # The ham row must remain an LLM rewrite.
    ham_row = written.loc[written["label"] == 0].iloc[0]
    assert ham_row["rewrite_method"] == REWRITE_METHOD
    assert "lunch" in ham_row["text"].lower()
