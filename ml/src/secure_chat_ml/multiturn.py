"""Build synthetic multi-turn windows that mirror what the browser actually scores.

The app feeds the model up to six recent turns as "Me: ..." / "Them: ..." lines with the
current message last (see frontend/src/ml/conversationContext.ts). The classifiers were
trained on single messages, so this module recreates that serving format for evaluation
and for augmenting distillation training data (train/serve skew fix).
"""

# Import random for seeded, reproducible window construction.
import random

# Import pandas for the frame helpers.
import pandas as pd

# Match CONTEXT_WINDOW_MESSAGES in conversationContext.ts (current message + 5 priors).
CONTEXT_WINDOW_MESSAGES = 6
# Match CONTEXT_MAX_CHARS in conversationContext.ts (oldest turns dropped beyond this).
CONTEXT_MAX_CHARS = 900


def format_turn(speaker: str, text: str) -> str:
    """Format one turn exactly like formatConversationTurn in the browser."""

    # Speaker is "Me" or "Them"; the colon-space separator matches the TypeScript.
    return f"{speaker}: {text}"


def join_keeping_current(turns: list[str], max_chars: int = CONTEXT_MAX_CHARS) -> str:
    """Drop the oldest prior turns until the window fits, always keeping the last turn."""

    # The last line is the message being scored and is never dropped.
    current = turns[-1]
    prior = turns[:-1]
    # Remove from the front while the joined text is too long.
    while prior and len("\n".join([*prior, current])) > max_chars:
        prior = prior[1:]
    # Join with newlines like the browser.
    return "\n".join([*prior, current])


def wrap_in_window(
    text: str,
    prior_texts: list[str],
    rng: random.Random,
    *,
    current_speaker: str = "Them",
    window: int = CONTEXT_WINDOW_MESSAGES,
) -> str:
    """Wrap `text` as the final turn after up to window-1 benign prior turns."""

    # Keep only as many priors as the production window allows.
    priors = prior_texts[-(max(1, window) - 1) :]
    # Alternate speakers starting from a random side so both orders appear.
    speaker = rng.choice(["Me", "Them"])
    lines: list[str] = []
    for prior in priors:
        lines.append(format_turn(speaker, prior))
        speaker = "Them" if speaker == "Me" else "Me"
    # The scored message comes last.
    lines.append(format_turn(current_speaker, text))
    return join_keeping_current(lines)


def build_multiturn_frame(
    frame: pd.DataFrame,
    benign_pool: list[str],
    *,
    seed: int = 42,
    max_priors: int = CONTEXT_WINDOW_MESSAGES - 1,
    fraction: float = 1.0,
) -> pd.DataFrame:
    """Return a copy of `frame` whose `text` is a windowed conversation.

    `benign_pool` must contain only legitimate messages so the context never carries
    the label by itself. `fraction` wraps only that share of rows (the rest stay
    single-message), which is how the distillation augmentation uses it.
    """

    # A separate RNG keeps this independent of any model seed.
    rng = random.Random(seed)
    out = frame.copy().reset_index(drop=True)
    texts: list[str] = []
    for text in out["text"].astype(str):
        # Leave some rows untouched when fraction < 1.
        if rng.random() > fraction or not benign_pool:
            texts.append(text)
            continue
        # Random context length, including zero priors (a first message).
        count = rng.randint(0, max_priors)
        priors = [rng.choice(benign_pool) for _ in range(count)]
        texts.append(wrap_in_window(text, priors, rng))
    out["text"] = texts
    return out
