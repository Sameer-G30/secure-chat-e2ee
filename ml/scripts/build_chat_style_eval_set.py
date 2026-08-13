"""Build the hand-curated, chat-style, evaluation-only scam/legitimate dataset.

Every source dataset in `data/processed/` is email or SMS register text
(headers, boilerplate, formal phrasing). None of it looks like an actual
DM/chat message. This script writes out a small (~80-row), manually
authored set of short, informal, chat-style messages — the kind a person
would actually receive in WhatsApp/iMessage/Discord/Instagram DMs — so the
baseline (and later DistilBERT) model's real-world behavior can be sanity
checked outside its training register.

Per `data/label-schema.yaml`'s `evaluation_policy`, this set is
evaluation-only: `scripts/evaluate_chat_style_eval.py` only ever calls
`pipeline.predict(...)` on it, never `pipeline.fit(...)`. Every row below
was written or reviewed by a human for this project; none of it is scraped
real user data.

Usage (from ml/):
    uv run python scripts/build_chat_style_eval_set.py
"""

# Import csv to write a schema-shaped CSV without hand-escaping commas/quotes.
import csv

# Import Path for a portable output location.
from pathlib import Path

_OUTPUT_PATH = Path("data/chat_eval/chat_style_eval_v1.csv")

# Label chat-style messages that read as ordinary, harmless conversation.
LEGITIMATE_LABEL = 0
# Label chat-style messages that read as a scam, phishing, or fraud attempt.
SCAM_LABEL = 1

# Hand-authored legitimate chat-style messages: short, informal, varied topics.
_LEGITIMATE_MESSAGES = [
    "hey are we still on for lunch tomorrow?",
    "omg yes that movie was so good, we need to talk about the ending",
    "can u send me the notes from today's lecture",
    "running 5 mins late, save me a seat",
    "happy bday!! hope ur having an amazing day 🎂",
    "did you finish the assignment yet lol im stuck on q3",
    "lol that meme you sent literally killed me",
    "mom said dinner's at 7, don't be late",
    "can you pick up milk on your way home",
    "just landed! will call you in a bit",
    "wyd this weekend, wanna hit the new cafe?",
    "thanks for helping me move all that furniture yesterday",
    "the wifi here is so bad lmao",
    "sent you the photos from the trip, check ur email",
    "bro I'm so tired, exam week is killing me",
    "let's reschedule our call to friday, something came up",
    "congrats on the new job!! so proud of you",
    "can you send me your address again, forgot to save it",
    "the game starts at 8, don't forget to bring snacks",
    "ugh traffic is insane rn, gonna be like 20 mins late",
    "hey did you get the group project doc I shared",
    "haha yeah I saw that tweet too, so cursed",
    "we still meeting at the library at 3?",
    "loved the playlist you made, adding it to my library",
    "can u proofread my email before I send it to the professor",
    "just got back from the gym, absolutely dead",
    "what time does the store close today",
    "sending good vibes for your interview tomorrow!",
    "my phone's at 2% gonna call you back later",
    "thank you so much for the birthday gift, I love it",
    "can we push our study session to sunday instead",
    "did you watch the new episode yet, no spoilers pls",
    "grabbing coffee, want me to bring you one",
    "the concert tickets went on sale, wanna go together",
    "sorry for the late reply, been swamped with work",
    "hope your flight lands safely, text me when you land",
    "can you feed the cat while I'm away this weekend",
    "that recipe you sent was amazing, making it again tonight",
    "let me know when you're free to video call",
    "good luck on your exam today, you got this",
]

# Hand-authored scam/phishing chat-style messages, mirroring real DM scam patterns.
_SCAM_MESSAGES = [
    "hey it's me, I lost my phone, can you send money to this new number urgently",
    "OMG i just made $3000 in a week trading crypto, message me and I'll show you how",
    "Your Netflix payment failed, update your card here: bit.ly/netflix-billing-fix",
    "You've been selected for a $500 Amazon gift card! Claim now: giftcard-verify.co/claim",
    "hi, I'm a talent recruiter, we'd love to hire you, just send your bank details to set up payroll",
    "URGENT: someone tried logging into your Instagram, verify your password now at instasecure-login.com",
    "Congratulations!! You won our weekly giveaway, DM your card number to receive the prize",
    "hey bestie, I'm stuck abroad and need $200 wired asap, will pay you back I promise",
    "Your package couldn't be delivered, pay a small $1.99 customs fee here: track-delivery-usps.info",
    "I'm a widow with $2M inheritance, I need a trustworthy partner to help transfer funds, interested?",
    "Your bank flagged suspicious activity, click here to confirm it's you before your account locks",
    "hey it's your bank, we need your OTP to verify a recent transaction, reply with the code now",
    "I can double any crypto you send me within 24 hours, guaranteed, just send to this wallet",
    "This is IT support, we detected a virus on your device, download this tool to fix it immediately",
    "you left your card at the store, send your card number so we can hold it for you",
    "Hi love, I feel such a connection with you, can you send me itunes gift cards so we can talk more",
    "Final notice: your electricity will be shut off today unless you pay via this link right now",
    "I'm your new boss, I need you to buy $300 in gift cards for a client meeting, keep receipts",
    "Click this link to see who viewed your profile 👀 freeprofileviews-check.net",
    "Your account has unusual sign-in activity, verify immediately or it will be permanently suspended",
    "I work for the IRS, you owe back taxes, pay now via gift card or a warrant will be issued",
    "hey I'm selling my old laptop cheap, just send a deposit via this payment link first",
    "You're pre-approved for a $10,000 loan, no credit check, just confirm your SSN to proceed",
    "Scan this QR code to claim your free reward before it expires in 10 minutes",
    "This is WhatsApp support, your account will be deleted unless you verify your number here",
    "I invested with this broker and tripled my savings, DM me and I'll get you set up too",
    "Your subscription renewal failed, re-enter your card details here to avoid service interruption",
    "hi, this is your delivery driver, I need you to pay an extra fee through this app to release your parcel",
    "we matched on the app, I really like you, can you help me pay my rent this month, I'll pay you back",
    "Your social security number has been suspended due to suspicious activity, call this number now",
    "you've inherited $1,000,000 from a relative you didn't know, send processing fee to claim",
    "this investment group guarantees 40% weekly returns, spots filling fast, send your deposit today",
    "Someone is trying to reset your password, if this wasn't you enter your current password here to stop it",
    "hey it's grandma, I'm in trouble and need bail money wired right away, please don't tell anyone",
    "Your PayPal has been limited, verify your identity within 24 hours: paypal-account-check.net",
    "I'm a model scout, send your bank info so we can pay your modeling fee in advance",
    "your device has a critical security issue, call this toll-free number immediately to fix it",
    "You qualify for student loan forgiveness, just confirm your bank account to receive the credit",
    "I'm stuck at customs and they want a bribe to release my luggage, can you send $150 now",
    "verify your crypto wallet now or your funds will be frozen permanently, link below",
]


def main() -> None:
    """Write the combined, shuffled-by-construction chat-style eval CSV."""

    rows = []
    for index, text in enumerate(_LEGITIMATE_MESSAGES):
        rows.append(
            {
                "message_id": f"chat-eval-legit-{index:03d}",
                "text": text,
                "label": LEGITIMATE_LABEL,
                "original_label": "legitimate_chat",
                "source": "chat_style_eval_v1",
                "split": "eval_only",
            }
        )
    for index, text in enumerate(_SCAM_MESSAGES):
        rows.append(
            {
                "message_id": f"chat-eval-scam-{index:03d}",
                "text": text,
                "label": SCAM_LABEL,
                "original_label": "scam_chat",
                "source": "chat_style_eval_v1",
                "split": "eval_only",
            }
        )

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["message_id", "text", "label", "original_label", "source", "split"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Wrote {len(rows)} rows "
        f"({len(_LEGITIMATE_MESSAGES)} legitimate, {len(_SCAM_MESSAGES)} scam) to {_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
