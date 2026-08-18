"""Build the hand-curated, chat-style, evaluation-only scam/legitimate dataset.

Every source dataset in `data/processed/` (and the rewritten chat-register
training text in `data/processed_chat/`) is derived from email or SMS.
None of that training text is this locked file. This script writes 200
hand-authored, informal, WhatsApp/iMessage/DM-style messages so the
baseline can be scored out of domain.

Per `data/label-schema.yaml`'s `evaluation_policy`, this set is
evaluation-only: `scripts/evaluate_chat_style_eval.py` only ever calls
`pipeline.predict(...)` on it, never `pipeline.fit(...)`, and never
retunes a threshold on it. These rows are never rewritten into
`data/processed_chat/`. Every row below was written or reviewed by a
human for this project; none of it is scraped real user data.

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

# Expected locked-set size: 100 legitimate + 100 scam = 200 rows.
_EXPECTED_PER_CLASS = 100

# Hand-authored legitimate chat-style messages: short, informal, varied topics.
# Includes ordinary shared https links so "has a URL" is not treated as scam.
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
    "can you review the pr when you get a sec",
    "i'm outside, should i park in the back lot",
    "we still on for tennis at 6",
    "my laptop charger died, you have an extra",
    "sending the zoom link for class https://zoom.us/j/5551112222",
    "here's the shared notes https://docs.google.com/document/d/1abcNotesDemo",
    "map to the restaurant https://maps.google.com/?q=Oak+Street+Bistro",
    "the github readme is here https://github.com/example/notes#readme",
    "mom's flight lands at 4:10, can you pick her up",
    "i'll bring dessert if you handle mains",
    "that article you sent was actually really good",
    "do we need a reservation or is it walk-in",
    "class got moved to room 204 btw",
    "i left your jacket in my car, grabbing it later",
    "wanna watch the match at my place",
    "coffee tomorrow before standup?",
    "the printer on 3rd floor is working again",
    "thanks for covering my shift yesterday",
    "i booked saturday 2pm for the haircut",
    "lol i just saw your story, the dog is so cute",
    "i'm skipping the gym today, back tomorrow",
    "reminder: rent is due friday, i already sent my half",
    "here's the recipe https://www.allrecipes.com/recipe/22764/potato-soup",
    "meeting notes are in the usual folder",
    "can we swap tuesdays for the carpool",
    "i finished the slides, tell me if slide 4 is too busy",
    "caught the bus, there in 15",
    "your package arrived, i signed for it",
    "we should try that new ramen place",
    "i'm making pasta, you want some leftover",
    "prof posted grades, check the portal",
    "i'll be offline on a hike till sunday",
    "bring a jacket, it's windy at the lake",
    "did you see the group chat about saturday",
    "i can take notes if you lead the discussion",
    "the museum is free this thursday",
    "need your opinion on these two paint colors",
    "i set an alarm, don't let me sleep past 8",
    "sending you the track i mentioned",
    "we still have milk, just need eggs",
    "my cousin is in town, dinner friday?",
    "that lecture made way more sense today",
    "i'll upload the photos tonight",
    "can you water the plants too not just the cat",
    "i found a cheaper textbook edition",
    "let's leave by 7 to beat traffic",
    "the trailer looks fun, we should go opening week",
    "i saved you a bagel in the kitchen",
    "your umbrella is still in my backpack",
    "study room 3 is free at 2 if we grab it",
    "i voted for the pizza place in the poll",
    "thanks, that explanation actually clicked",
    "i'm on the train, call you when i get off",
    "we need more chairs if 8 people are coming",
    "here's the public docs https://support.google.com/docs",
    "the weather app says rain after 5",
    "i'll handle dishes if you cook",
    "just confirming 3pm at the library still works",
    "office wifi is guest-office, password is on the fridge",
    "putting the extra tickets in your mailbox later",
]

# Hand-authored scam/phishing chat-style messages, mirroring real DM scam patterns.
# Wording is independent of the rule_based_v1 rewriter templates.
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
    "your icloud storage is full, confirm billing at appleid-secure-billing.net/login",
    "we need your seed phrase to sync the wallet, reply with the 12 words",
    "this airdrop ends in 20 minutes, connect wallet at claim-drop.example.xyz",
    "i can get you verified on the exchange, just send a selfie holding your id and a code",
    "customs held your parcel, unlock with a card payment at parcel-unlock.top/pay",
    "your cousin anna changed numbers, send 400 for the hospital bill today",
    "hr payroll update, reply with routing number and ssn to keep getting paid",
    "police warrant in your name, pay the processing fee with bitcoin now",
    "your steam inventory will be reset, log in at steamcommunnity-login.com",
    "i'm locked out of my exchange, can you receive my crypto then send it back",
    "kids school trip fee, pay this new vendor link before noon",
    "your domain expires tonight, renew with card at domain-renew-now.click",
    "we found your car on camera, pay the fine here to avoid towing",
    "send amazon cards so i can join the video app they require for dating",
    "your 2fa backup codes leaked, confirm them here to rotate",
    "tiktok copyright strike, appeal at tiktok-appeals-help.io/verify",
    "i already paid the courier, just refund me to this cashapp",
    "your nft whitelist spot, mint from this contract before gas spikes",
    "bank: unusual venmo, reply YES and your pin to approve",
    "remote job, buy the laptop from our partner link and we'll reimburse",
    "your recovery email was changed, undo at account-restore-secure.com",
    "doctor office: update insurance, enter member id and ssn at this portal",
    "i'll send you $2000 extra, deposit $200 first to unlock the transfer",
    "whatsapp new terms, verify or your chats get deleted tonight",
    "your neighbor's wifi cam saw a package thief, pay to see the clip",
    "crypto recovery specialist, we need remote access software now",
    "you won concert tickets, pay shipping with a gift card code",
    "charity for the disaster, only accept crypto to this address",
    "your driving license points, settle the fine at gov-pay-ticket.xyz",
    "i'm at the airport and my card declined, apple cash 180 please",
    "microsoft 365 admin, your tenant is compromised, share screen",
    "we mirrored your wallet, sign this typed data to revoke the thief",
    "your pet insurance lapsed, reactivate with card at petsure-pay.info",
    "onlyfans leak warning, pay removal fee here",
    "family group: dad's in a meeting, send the code from the bank sms to me",
    "your airline booking needs extra baggage fee at airline-pay-extra.top",
    "kyc refresh for the broker, upload passport plus a utility bill now",
    "i'll trade you rare skins, first send the items i'll send payment after",
    "electric vehicle rebate, confirm bank details to receive it this week",
    "your icloud photos are public, lock them at apple-privacy-lock.com",
    "we need one more otp to stop a wire, text it back",
    "scholarship disbursement, pay a $49 clearance fee at grant-clear.info",
    "your cloud backup failed, re-enter password at backup-restore-now.net",
    "this invoice from your printer vendor is overdue, pay immediately at bill-pay.click",
    "your social media brand deal, we need your login to schedule posts",
    "refund waiting, confirm identity at tax-refund-check.xyz",
    "i'm a soldier overseas, need itunes cards to call family",
    "your home security system expires, renew at adt-billing-secure.com",
    "we duplicated your keys from a photo, pay to stop distribution",
    "binance support: freeze will lift after you send 0.2 eth to the safety wallet",
    "your child's school portal, confirm current password to keep grades visible",
    "limited beta of the bank app, install this apk and log in",
    "you were caught on a speeding cam, discount if you pay in 15 minutes",
    "we will list your house unless you verify owner details at title-check.top",
    "your paypal goods hold, upload card photos to release funds",
    "hey, can you authorize this new device, reply with the 6-digit code",
    "investment club weekly 25 percent, send usdt to stay in the pool",
    "your esim will deactivate, re-register passport at carrier-verify.xyz",
    "i have a video of you, pay btc or it goes to your contacts",
    "your mailbox is full, log in at https://192.0.2.55/webmail/login",
]


# Build schema-shaped rows from the two hand-authored lists.
def _rows() -> list[dict[str, str | int]]:
    """Return 200 labeled chat-eval rows in schema column order."""

    # Fail in the builder if a future edit changes the locked-set size by accident.
    if len(_LEGITIMATE_MESSAGES) != _EXPECTED_PER_CLASS:
        # Keep the locked set at 100 legitimate rows.
        raise ValueError(
            f"expected {_EXPECTED_PER_CLASS} legitimate messages, "
            f"got {len(_LEGITIMATE_MESSAGES)}"
        )
    # Fail if the scam list is not exactly 100 unique-intent rows.
    if len(_SCAM_MESSAGES) != _EXPECTED_PER_CLASS:
        # Keep the locked set at 100 scam rows.
        raise ValueError(
            f"expected {_EXPECTED_PER_CLASS} scam messages, got {len(_SCAM_MESSAGES)}"
        )
    # Reject accidental duplicate strings inside a class.
    if len(set(_LEGITIMATE_MESSAGES)) != len(_LEGITIMATE_MESSAGES):
        # Duplicate legitimate wording would shrink the effective eval set.
        raise ValueError("duplicate legitimate chat-eval messages")
    # Reject accidental duplicate strings inside the scam class.
    if len(set(_SCAM_MESSAGES)) != len(_SCAM_MESSAGES):
        # Duplicate scam wording would shrink the effective eval set.
        raise ValueError("duplicate scam chat-eval messages")
    # Reject a message that appears in both classes.
    if set(_LEGITIMATE_MESSAGES) & set(_SCAM_MESSAGES):
        # A row cannot be both legitimate and scam.
        raise ValueError("chat-eval message appears in both classes")
    # Accumulate schema-shaped dictionaries for csv.DictWriter.
    rows: list[dict[str, str | int]] = []
    # Assign stable identifiers to the legitimate half.
    for index, text in enumerate(_LEGITIMATE_MESSAGES):
        # Append one locked legitimate eval row.
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
    # Assign stable identifiers to the scam half.
    for index, text in enumerate(_SCAM_MESSAGES):
        # Append one locked scam eval row.
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
    # Return the combined 200-row list.
    return rows


def main() -> None:
    """Write the combined chat-style eval CSV (evaluation-only, never for training)."""

    # Build and validate the 200 locked rows.
    rows = _rows()
    # Create data/chat_eval/ without failing when it already exists.
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Write a schema-shaped CSV the loaders already know how to read.
    with _OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        # Use the unified label-schema column order.
        writer = csv.DictWriter(
            handle, fieldnames=["message_id", "text", "label", "original_label", "source", "split"]
        )
        # Write the header row first.
        writer.writeheader()
        # Write every locked evaluation row.
        writer.writerows(rows)

    # Report only counts and the path, never the message content.
    print(
        f"Wrote {len(rows)} rows "
        f"({len(_LEGITIMATE_MESSAGES)} legitimate, {len(_SCAM_MESSAGES)} scam) to {_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
