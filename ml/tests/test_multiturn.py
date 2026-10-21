"""Tests for synthetic multi-turn windows."""

import random

import pandas as pd

from secure_chat_ml.multiturn import (
    CONTEXT_MAX_CHARS,
    build_multiturn_frame,
    join_keeping_current,
    wrap_in_window,
)


def test_current_message_is_always_last_and_kept() -> None:
    rng = random.Random(0)
    window = wrap_in_window("click http://x.co now", ["hi"] * 5, rng)
    assert window.splitlines()[-1].endswith("click http://x.co now")


def test_window_respects_char_budget_by_dropping_oldest() -> None:
    long_turns = ["Me: " + "a" * 400, "Them: " + "b" * 400, "Me: " + "c" * 400, "Them: scam"]
    joined = join_keeping_current(long_turns)
    assert len(joined) <= CONTEXT_MAX_CHARS
    assert joined.endswith("Them: scam")


def test_frame_keeps_labels_and_length() -> None:
    frame = pd.DataFrame({"text": ["a", "b", "c"], "label": [0, 1, 1]})
    out = build_multiturn_frame(frame, ["hello", "ok"], seed=1)
    assert list(out["label"]) == [0, 1, 1]
    assert len(out) == 3
    # Deterministic for a fixed seed.
    again = build_multiturn_frame(frame, ["hello", "ok"], seed=1)
    assert list(out["text"]) == list(again["text"])
