// Unit tests for the ChatScreen conversation-window scoring policy.

// Import Vitest helpers used by the rest of the frontend unit suite.
import { describe, expect, it } from 'vitest'

// Import the policy helpers under test.
import {
  applyBenignLinkBannerPolicy,
  applyChatScreenThreshold,
  buildConversationScoringText,
  CHATSCREEN_MIN_SCAM_THRESHOLD,
  collectPriorTurns,
  CONTEXT_WINDOW_MESSAGES,
  isBenignLinkTurn,
  isTrivialShortNoUrl,
  joinTurnsKeepingCurrent,
  messageHasUrl,
  SHORT_BENIGN_LINK_PROSE_CHARS,
  TRIVIAL_MAX_CHARS,
  windowHasScamPattern,
} from './conversationContext'
import type { ConversationTurn } from './conversationContext'

// Build a verified turn with the fields the window actually reads.
function turn(
  id: string,
  direction: ConversationTurn['direction'],
  plaintext: string,
): ConversationTurn {
  // verificationFailed stays false so the turn is eligible for concatenation.
  return { id, direction, plaintext, verificationFailed: false }
}

// Describe skip / URL / window / threshold behavior without loading ORT.
describe('conversationContext', () => {
  // Pin the product constants so a silent change cannot widen the skip.
  it('keeps a 3 message window and an 8-character trivial skip', () => {
    // The current message plus the two turns before it.
    expect(CONTEXT_WINDOW_MESSAGES).toBe(3)
    // Eight characters skips "ok"/"thanks" but not "click here".
    expect(TRIVIAL_MAX_CHARS).toBe(8)
    // ChatScreen overlay sits above tfidf_best's frozen 0.20 cut.
    expect(CHATSCREEN_MIN_SCAM_THRESHOLD).toBe(0.35)
  })

  // Skip only tiny URL-free chit-chat; everything else still reaches the model.
  it('skips only trivial short messages that have no URL', () => {
    // Empty / whitespace-only lines are skipped.
    expect(isTrivialShortNoUrl('')).toBe(true)
    expect(isTrivialShortNoUrl('   ')).toBe(true)
    // Classic chit-chat under the cap, no URL, no digits.
    expect(isTrivialShortNoUrl('ok')).toBe(true)
    expect(isTrivialShortNoUrl('yes')).toBe(true)
    expect(isTrivialShortNoUrl('lol')).toBe(true)
    expect(isTrivialShortNoUrl('thanks')).toBe(true)
    expect(isTrivialShortNoUrl('will do')).toBe(true)
    // Longer greetings and split-scam openers still score.
    expect(isTrivialShortNoUrl('hello from bob')).toBe(false)
    expect(isTrivialShortNoUrl("hey it's me")).toBe(false)
    expect(isTrivialShortNoUrl('click here')).toBe(false)
    // Digits keep short money / OTP lines in the scoring path.
    expect(isTrivialShortNoUrl('send 500')).toBe(false)
    expect(isTrivialShortNoUrl('otp 12')).toBe(false)
    // A URL, even on an otherwise tiny line, is never skipped.
    expect(isTrivialShortNoUrl('https://bit.ly/x')).toBe(false)
    expect(messageHasUrl('see https://bit.ly/abc123 now')).toBe(true)
    expect(messageHasUrl('ok')).toBe(false)
  })

  // Prior turns stop at the current id and drop verification failures.
  it('collects verified prior turns and stops at the current message', () => {
    // Mix sent, received, a failed row, and the current DM.
    const transcript: ConversationTurn[] = [
      turn('1', 'received', 'hey'),
      { id: 'bad', direction: 'received', plaintext: null, verificationFailed: true },
      turn('2', 'sent', 'what is it'),
      turn('3', 'received', 'send this now'),
      turn('4', 'sent', 'ok'),
    ]
    // Scoring message 3 should see only hey + what is it.
    const prior = collectPriorTurns(transcript, '3')
    // Failed verification must not appear.
    expect(prior.map((row) => row.id)).toEqual(['1', '2'])
    // A send that is not yet in the transcript uses the whole list as prior.
    expect(collectPriorTurns(transcript, 'missing').map((row) => row.id)).toEqual(['1', '2', '3', '4'])
  })

  // Concatenate last 3 turns with speaker labels; drop the older turns.
  it('builds a speaker-labeled window of the last 3 messages', () => {
    // Seven verified turns so everything before the last two priors must fall out.
    const transcript: ConversationTurn[] = [
      turn('1', 'received', 'one'),
      turn('2', 'sent', 'two'),
      turn('3', 'received', 'three'),
      turn('4', 'sent', 'four'),
      turn('5', 'received', 'five'),
      turn('6', 'sent', 'six'),
      turn('7', 'received', 'seven current'),
    ]
    // Score the last DM against the open transcript.
    const text = buildConversationScoringText(transcript, '7', 'seven current', 'received')
    // Turns older than the two priors are outside the window of 3.
    expect(text).not.toContain('one')
    expect(text).not.toContain('two')
    expect(text).not.toContain('three')
    expect(text).not.toContain('four')
    // The two priors plus the current message stay, with speaker labels.
    expect(text).toContain('Them: five')
    expect(text).toContain('Me: six')
    expect(text).toContain('Them: seven current')
    // Current DM stays last so left-truncation cannot drop it first.
    expect(text.endsWith('Them: seven current')).toBe(true)
  })

  // Character budget drops oldest formatted lines, never the current DM.
  it('left-trims prior turns when the concatenated window is over budget', () => {
    // Two long priors plus a short current line.
    const joined = joinTurnsKeepingCurrent(
      ['Them: ' + 'a'.repeat(80), 'Me: ' + 'b'.repeat(80), 'Them: now'],
      40,
    )
    // The current line must survive even when priors do not fit.
    expect(joined).toBe('Them: now')
  })

  // ChatScreen overlay raises 0.20/0.30 frozen cuts without lowering a stricter one.
  it('applies the 0.35 ChatScreen overlay on top of the frozen sidecar cut', () => {
    // TF-IDF Best frozen 0.20 would have warned at 0.25; ChatScreen must not.
    expect(
      applyChatScreenThreshold({
        pScam: 0.25,
        warned: true,
        checkpointId: 'tfidf_best',
        inferenceMs: 1,
      }).warned,
    ).toBe(false)
    // 0.40 clears both the frozen cut and the overlay.
    expect(
      applyChatScreenThreshold({
        pScam: 0.4,
        warned: true,
        checkpointId: 'tfidf_best',
        inferenceMs: 1,
      }).warned,
    ).toBe(true)
    // A stricter frozen cut (warned already false) must stay unwarned.
    expect(
      applyChatScreenThreshold({
        pScam: 0.4,
        warned: false,
        checkpointId: 'tfidf_default',
        inferenceMs: 1,
      }).warned,
    ).toBe(false)
  })

  // A short well-known link must not banner from the URL alone. Thresholds stay put.
  it('does not banner a normal sentence with a Google Docs link from the URL alone', () => {
    // The prose cap stays a short line, not a paragraph.
    expect(SHORT_BENIGN_LINK_PROSE_CHARS).toBe(120)
    // The frozen overlay cut is unchanged.
    expect(CHATSCREEN_MIN_SCAM_THRESHOLD).toBe(0.35)
    // Fixture the student scores near zero in isolation, and high when the URL is bare.
    const current = 'here is the shared doc https://docs.google.com/document/d/abc123'
    // The window is that one turn. No account-locked or shortener context.
    const windowText = `Them: ${current}`
    // The bubble qualifies as a short well-known link.
    expect(isBenignLinkTurn(current)).toBe(true)
    // The window has no listed scam pattern.
    expect(windowHasScamPattern(windowText)).toBe(false)
    // Even a high model score must not raise the banner from this URL alone.
    const adjusted = applyBenignLinkBannerPolicy(
      { pScam: 0.91, warned: true, checkpointId: 'distilbert_student', inferenceMs: 1 },
      current,
      windowText,
    )
    // Banner off. The probability and the frozen threshold are not rewritten.
    expect(adjusted.warned).toBe(false)
    expect(adjusted.pScam).toBe(0.91)
    // A bare docs URL is the same exception.
    expect(isBenignLinkTurn('https://docs.google.com/document/d/abc123')).toBe(true)
    // Drive, GitHub, and YouTube are the same kind of host.
    expect(isBenignLinkTurn('https://drive.google.com/file/d/abc')).toBe(true)
    expect(isBenignLinkTurn('notes https://github.com/org/repo')).toBe(true)
    expect(isBenignLinkTurn('https://www.youtube.com/watch?v=abc123')).toBe(true)
  })

  // Scam context and shorteners still banner. The model score is left in place.
  it('still banners verify-immediately, gift cards, account locked, and bit.ly', () => {
    // The shortener line the corpora taught the model to flag.
    const shortener = 'verify immediately https://bit.ly/phish'
    // That bubble is not a well-known-host exception.
    expect(isBenignLinkTurn(shortener)).toBe(false)
    // The window contains both the phrase and the shortener.
    expect(windowHasScamPattern(`Them: ${shortener}`)).toBe(true)
    // A warned model result stays warned.
    expect(
      applyBenignLinkBannerPolicy(
        { pScam: 0.88, warned: true, checkpointId: 'distilbert_student', inferenceMs: 1 },
        shortener,
        `Them: ${shortener}`,
      ).warned,
    ).toBe(true)
    // A bare shortener is not excused.
    expect(isBenignLinkTurn('https://bit.ly/phish')).toBe(false)
    // A docs link after an earlier locked-account turn still banners.
    const docs = 'https://docs.google.com/document/d/abc123'
    const lockedWindow = `Them: your account is locked\nMe: ${docs}`
    expect(isBenignLinkTurn(docs)).toBe(true)
    expect(windowHasScamPattern(lockedWindow)).toBe(true)
    expect(
      applyBenignLinkBannerPolicy(
        { pScam: 0.8, warned: true, checkpointId: 'distilbert_best', inferenceMs: 1 },
        docs,
        lockedWindow,
      ).warned,
    ).toBe(true)
    // Gift-card context in the window keeps the banner too.
    expect(windowHasScamPattern('Them: buy two gift cards\nMe: https://drive.google.com/file/d/abc')).toBe(
      true,
    )
    // javascript: is not a benign http(s) link, so the policy does not clear a banner.
    expect(isBenignLinkTurn('javascript:alert(1)')).toBe(false)
    // A long note is normal text. The three-turn score stands.
    const paragraph = `${'Please read the notes from yesterday. '.repeat(8)}https://docs.google.com/document/d/abc123`
    expect(isBenignLinkTurn(paragraph)).toBe(false)
    expect(
      applyBenignLinkBannerPolicy(
        { pScam: 0.7, warned: true, checkpointId: 'distilbert_student', inferenceMs: 1 },
        paragraph,
        `Them: ${paragraph}`,
      ).warned,
    ).toBe(true)
  })
})
