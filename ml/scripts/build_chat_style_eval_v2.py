"""Build chat_style_eval_v2.csv, a development-only OOD set.

chat_style_eval_v1.csv stays locked for the final reported number. This v2 file
is the set cascade/ensemble/length-filter decisions may look at. It is still
never used for Pipeline.fit or threshold search on TRAIN. Same schema as v1.

Usage (from ml/):
    uv run python scripts/build_chat_style_eval_v2.py
"""

# Import csv to write a schema-shaped CSV without hand-escaping.
import csv

# Import Path for a portable output location.
from pathlib import Path

# Import pandas to load the locked v1 file for an overlap check (never for fitting).
import pandas as pd

_OUTPUT_PATH = Path("data/chat_eval/chat_style_eval_v2.csv")
_V1_PATH = Path("data/chat_eval/chat_style_eval_v1.csv")
LEGITIMATE_LABEL = 0
SCAM_LABEL = 1
_EXPECTED_PER_CLASS = 100

# Hand-authored legitimate DMs, disjoint from v1. Ordinary links included.
_LEGITIMATE_MESSAGES = [
    "can you send the wifi password for the guest network",
    "i parked behind the blue civic, is that okay",
    "the lab is locked, meeting in the courtyard instead",
    "bringing sparkling water not soda is that fine",
    "did the professor post the rubric yet",
    "i'll take the early train so we can grab seats together",
    "your plants look thirsty, i watered the basil only",
    "movie starts 7:40, i'll meet you by the popcorn",
    "need a ride to the airport saturday 6am, can pay gas",
    "left my charger in seminar room 12 if you go back",
    "the group chat renamed itself to thesis panic again",
    "i found a cheaper parking garage two blocks south",
    "can we do laundry tomorrow instead of tonight",
    "the dog walker came, treats are on the counter",
    "i submitted the form, they said 5 business days",
    "want me to pick up your dry cleaning too",
    "the bookstore has used copies of chapter 4",
    "i'm making soup, tell me if you have a food allergy",
    "class cancelled, extra office hours thursday 2-4",
    "i screenshot the bus schedule, check your messages",
    "we still have the museum tickets for sunday right",
    "i'll bring folding chairs, you bring the speaker",
    "the printer jammed, i reset it, should be good",
    "can you confirm the babysitter's number",
    "i put your mail on the kitchen table",
    "the hike is 4 miles, bring extra water",
    "i liked that podcast rec, starting episode 2 tonight",
    "we can split the grocery bill on venmo later, no rush",
    "the library holds expire friday, i'll grab both books",
    "i'm skipping dessert, save me a corner piece if you want",
    "the zoom host changed, new link in the calendar invite",
    "i found your scarf in the lecture hall lost and found",
    "can we move brunch to 11, kitchen is slammed at 10",
    "i'll take notes in the shared doc during the call",
    "the bike shop said the tube will be ready at 5",
    "your mom called the landline, i said you'd ring back",
    "i booked the study room under my id, 6-8pm",
    "the weather is nicer on the east trail according to the ranger",
    "i can drive if you navigate, parking is confusing there",
    "we still need name tags for the volunteers",
    "i uploaded the photos to the shared album, not email",
    "the cafe takes cash only after 8, heads up",
    "i'll text when i leave campus so you can start the kettle",
    "the syllabus quiz is optional, i'm still doing it",
    "found a window seat, saving one with my backpack",
    "can you proof the caption before i post the event flyer",
    "the recycling bin is full, i'll take it down after dinner",
    "i set a reminder for the vaccine appointment thursday",
    "the concert merch line is short if we go now",
    "i'll meet you at exit 2 not the main doors",
    "the ta said drafts can be late with a note, send it",
    "i grabbed oat milk because they were out of almond",
    "we can watch at my place, hdmi cable is already plugged in",
    "the campus clinic opens at 9, i'll get in line at 8:40",
    "i left the spare key with the neighbor on the left",
    "the group project poll closes tonight, vote please",
    "i'll bring a power strip, outlets are scarce in that room",
    "the bakery has the loaf you like on weekdays only",
    "can you send the tracking number when you see it",
    "i'm on hold with the registrar, will update you",
    "the park has a free concert, picnic blankets encouraged",
    "i washed the extra towels, they're in the dryer",
    "we should leave 20 minutes earlier, construction on 5th",
    "i starred the email with the attachment so it doesn't sink",
    "the roommate's cousin is visiting, extra person for dinner",
    "i'll take the later bus so we arrive together",
    "the ice rink has student night, $5 if we go before 8",
    "i labeled the leftovers, chicken is the glass container",
    "can you grab napkins, i have plates and cups",
    "the slides are 80% done, need your chart on slide 6",
    "i found street parking, walk toward the mural",
    "the gym is closing early for maintenance, tomorrow 6am",
    "i'll send the calendar invite so it shows on your phone",
    "the farmer market has strawberries if you want jam later",
    "we can do a voice note if typing on the bus is annoying",
    "i reserved a court at 7, they cancel if we're 10 min late",
    "the extension cord is in the hall closet behind the vacuum",
    "i told the waiter separate checks, they said it's fine",
    "the campus shuttle is every 12 minutes after 6",
    "i'll bring a sweater, the theater is freezing",
    "the assignment portal glitched, i emailed the ta a pdf",
    "can you water the succulent, skip the fern, it's dramatic",
    "i put your name on the pizza, they won't hand it to me",
    "the walking tour starts at the clock tower not the gate",
    "i'll take the trash out if you wash the pans",
    "the used bookstore will hold the novel until friday",
    "we still on for language exchange, cafe or library?",
    "i screenshotted the seating chart, you're in row c",
    "the laundry card has $4, enough for one more load",
    "i'll meet you after the recitation, usual bench",
    "the community fridge restocked yogurt, grab some if you want",
    "can you send the wifi qr, typing the password is painful",
    "i packed an extra granola bar in your bag for the trip",
    "the office moved to the 4th floor, same building",
    "i'll keep my ringer on in case the gate code changes",
    "the potluck list still needs a salad, i can do that",
    "we can split an uber, surge looks low right now",
    "i bookmarked the public hours https://www.loc.gov/visit/",
    "here's the bus map https://www.google.com/maps",
    "notes folder is https://docs.google.com/document/d/v2NotesOnly",
]

# Hand-authored scam DMs, disjoint from v1, covering similar families with new wording.
_SCAM_MESSAGES = [
    "sis it's me, new number, send $400 to this cashapp so i can get home",
    "i doubled my paycheck in 48 hours on this coin, i can add you to the private group",
    "your hulu bill bounced, update payment at hulu-billing-help.co/pay",
    "selected for a $750 target gift card, claim at target-rewards-now.net",
    "hr contractor here, wire your routing number so we can start payroll today",
    "instagram security: confirm password at ig-secure-check.com to keep the account",
    "you won the weekend raffle, reply with cvv to ship the prize",
    "crypto desk: your withdrawal is frozen until you send 0.15 btc to the vault address",
    "mom i need itunes cards for a school portal fee, don't tell dad",
    "fedex: customs fee due, pay at fedex-duty-pay.top or the package returns",
    "your icloud is locked, verify at appleid-restore-help.com with your password",
    "binance support ticket, we need seed phrase to reverse the unauthorized trade",
    "hey handsome, i'm stuck overseas, just need $200 in bitcoin for a ticket",
    "irs notice, pay the back tax with gift cards at tax-settle-now.xyz",
    "your steam account was limited, unlock at steamcommunity-login.help",
    "bank fraud team, reply YES and the sms code to cancel a $2,400 wire",
    "we've been watching you, pay 0.3 btc or the video goes to your class group",
    "kyc update for coinbase, upload passport plus selfie at coinbase-kyc-verify.top",
    "your robinhood is restricted, restore access by sending usdt to this wallet",
    "amazon: unusual sign-in, confirm card at amzn-account-review.co",
    "i'm your cousin at the embassy, they seized my passport, western union please",
    "netflix prize, you are winner number 3, pay shipping with a visa gift card",
    "microsoft detected viruses, call this number and allow remote access now",
    "your wallet was cloned, sign this message to revoke the attacker's allowance",
    "payroll glitch, we overpaid you, refund $180 to this account today",
    "whatsapp will delete your chats tonight unless you verify at wa-verify-now.com",
    "your neighbor's camera caught a break-in, unlock the clip for $29 at cam-pay.info",
    "nft mint whitelist, connect wallet at mint-safe-drop.live before gas spikes",
    "doctor portal: update ssn and insurance at clinic-pay-secure.net",
    "i'll send $3000, just deposit $250 first so the transfer clears",
    "your icloud photos leaked, lock them for a removal fee at photo-takedown.xyz",
    "binance freeze lifts after you move 0.2 eth into the safety wallet we name",
    "school portal: confirm current password to keep grades visible this term",
    "install this apk to keep your bank app working, then log in once",
    "speeding camera, pay the discount in 10 minutes at city-ticket-pay.top",
    "we will list your apartment unless you verify owner info at title-lock.help",
    "paypal goods hold, photograph both sides of your card to release funds",
    "hey can you authorize my new phone, reply with the 6-digit sms code",
    "investment pool 30 percent weekly, send usdt to remain a member",
    "your esim dies tonight, re-register passport at carrier-unlock.xyz",
    "mailbox full, log in at http://198.51.100.12/webmail to keep receiving mail",
    "discord mod: your server is reported, verify token at discord-safety.help",
    "tiktok shop payout, enter bank details at tt-payout-verify.co",
    "your icloud backup failed, re-enter apple id at backup-apple-fix.net",
    "charity for the flood, only crypto to this address counts as a donation",
    "i'm a soldier, need google play credit to call family this week",
    "home alarm expires, renew at adt-pay-secure.info with any card",
    "we copied your house keys from a photo, pay to halt distribution",
    "metamask support, seed phrase required to unstick the pending swap",
    "your child's tablet is locked, parental unlock fee at family-pin.top",
    "uber refund, confirm the last four of your card at uber-cashback.co",
    "linkedin recruiter, we hire remote, buy the laptop kit then we reimburse 2x",
    "your domain expires in 2 hours, renew at domain-renew-fast.xyz",
    "spotify family owner removed you, rejoin at spotify-family-fix.com",
    "we have compromising photos, 0.4 btc to this wallet in one hour",
    "dhl: pay the import vat at dhl-vat-collect.top so we can deliver",
    "your google ads account is paused, verify billing at ads-google-pay.help",
    "imessage: this is apple support, your id was used in another country, send the code",
    "airbnb host change, send a holding deposit to this cashapp to keep the booking",
    "your tesla app login expired, reconnect at tesla-account-sync.co",
    "crypto recovery expert, install anydesk so we can save the remaining coins",
    "you were chosen for a tesla giveaway, pay transport at tesla-winner-ship.net",
    "bank: unusual zelle, reply STOP and your pin to cancel",
    "your icloud was accessed in russia, reset at apple-security-desk.com",
    "onlyfans leak notice, pay the takedown invoice here",
    "family gc: dad's in a meeting, send me the bank otp so i can finish taxes",
    "airline extra bag fee, pay at airline-bag-pay.top or they dump the luggage",
    "broker kyc refresh, utility bill plus passport at broker-kyc-now.xyz",
    "i'll trade rare skins, you send first then i pay",
    "ev rebate, confirm routing numbers to receive it this week",
    "your photos are public, lock at apple-privacy-lock.net with your password",
    "we need one more otp to stop a same-day wire, text it back now",
    "scholarship fee $49 at grant-release.info or you lose the award",
    "cloud backup failed, password again at restore-cloud-now.com",
    "printer vendor invoice overdue, pay immediately at bill-click.pay",
    "brand deal, we need your social login to schedule the posts",
    "tax refund waiting, identity check at refund-claim.xyz",
    "your driving fine, settle at gov-pay-ticket.help with a prepaid card",
    "i'm at the airport, card declined, apple cash 220 please",
    "365 admin, tenant compromised, share screen and run this script",
    "we mirrored your wallet, sign typed data to revoke the thief",
    "pet insurance lapsed, card at petsure-bill.info to reactivate",
    "whatsapp new policy, verify or chats vanish at midnight",
    "neighbor cam, pay to see the thief clip at security-clip.pay",
    "remote job, purchase equipment from our partner then we repay",
    "recovery email changed, undo at account-restore-now.com",
    "courier already paid, refund me on cashapp this number",
    "nft whitelist, mint from this contract before the snapshot",
    "venmo unusual, reply YES plus pin to approve",
    "microsoft 365, your license lapses, login at office-renew-pay.top",
    "your seed phrase is required to stop a drain, paste it here",
    "hi mom it's me, i broke my phone, send money to this new wallet",
    "prize: $1,000 walmart card, confirm address at walmart-claim.co",
    "support: we are apple, your id is locked, read the code from sms to me",
    "your package is held, pay $1.99 at ups-hold-fee.xyz",
    "crypto airdrop, connect wallet and sign to receive 800 tokens",
    "i made $8k this week on options, join my telegram for signals, entry $99",
    "your facebook is disabled, appeal at fb-appeal-login.help",
    "bank otp needed to stop a hacker, tell me the 6 digits now",
    "your icloud storage is full, pay at apple-icloud-invoice.net or lose photos",
]


def _rows() -> list[dict[str, str | int]]:
    """Return 200 labeled v2 rows, refusing overlap with v1 or within-class dupes."""

    if len(_LEGITIMATE_MESSAGES) != _EXPECTED_PER_CLASS:
        raise ValueError(
            f"expected {_EXPECTED_PER_CLASS} legitimate messages, got {len(_LEGITIMATE_MESSAGES)}"
        )
    if len(_SCAM_MESSAGES) != _EXPECTED_PER_CLASS:
        raise ValueError(f"expected {_EXPECTED_PER_CLASS} scam messages, got {len(_SCAM_MESSAGES)}")
    if len(set(_LEGITIMATE_MESSAGES)) != len(_LEGITIMATE_MESSAGES):
        raise ValueError("duplicate legitimate v2 messages")
    if len(set(_SCAM_MESSAGES)) != len(_SCAM_MESSAGES):
        raise ValueError("duplicate scam v2 messages")
    if set(_LEGITIMATE_MESSAGES) & set(_SCAM_MESSAGES):
        raise ValueError("v2 message appears in both classes")
    if _V1_PATH.exists():
        v1_texts = set(pd.read_csv(_V1_PATH)["text"].astype(str).tolist())
        overlap = (set(_LEGITIMATE_MESSAGES) | set(_SCAM_MESSAGES)) & v1_texts
        if overlap:
            raise ValueError(f"v2 overlaps locked v1 on {len(overlap)} strings")
    rows: list[dict[str, str | int]] = []
    for index, text in enumerate(_LEGITIMATE_MESSAGES):
        rows.append(
            {
                "message_id": f"chat-eval-v2-legit-{index:03d}",
                "text": text,
                "label": LEGITIMATE_LABEL,
                "original_label": "legitimate_chat",
                "source": "chat_style_eval_v2",
                "split": "eval_only",
            }
        )
    for index, text in enumerate(_SCAM_MESSAGES):
        rows.append(
            {
                "message_id": f"chat-eval-v2-scam-{index:03d}",
                "text": text,
                "label": SCAM_LABEL,
                "original_label": "scam_chat",
                "source": "chat_style_eval_v2",
                "split": "eval_only",
            }
        )
    return rows


def main() -> None:
    """Write the development-only v2 CSV next to the locked v1 file."""

    rows = _rows()
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
