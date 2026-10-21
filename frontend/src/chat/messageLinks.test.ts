// Unit tests for http(s)-only link splitting. No network.

// Import Vitest helpers.
import { describe, expect, it } from 'vitest'

// Import the splitter under test.
import { messageHasHttpLink, splitHttpLinks } from './messageLinks'

// Describe scheme filtering and trailing punctuation.
describe('messageLinks', () => {
  // A normal sentence keeps the words and exposes one https anchor.
  it('links an https URL inside a sentence', () => {
    // The shared-doc fixture used by the scoring policy.
    const parts = splitHttpLinks('here is the shared doc https://docs.google.com/document/d/abc123')
    // Prose, then the link.
    expect(parts).toEqual([
      { kind: 'text', text: 'here is the shared doc ' },
      {
        kind: 'link',
        text: 'https://docs.google.com/document/d/abc123',
        href: 'https://docs.google.com/document/d/abc123',
      },
    ])
    // The bubble layout uses this to switch off the single-button body.
    expect(messageHasHttpLink('here is the shared doc https://docs.google.com/document/d/abc123')).toBe(
      true,
    )
  })

  // Trailing punctuation stays outside the href.
  it('leaves sentence punctuation outside the link', () => {
    // Period belongs to the sentence.
    const parts = splitHttpLinks('see https://github.com/org/repo.')
    // The link text has no period.
    expect(parts[1]).toEqual({
      kind: 'link',
      text: 'https://github.com/org/repo',
      href: 'https://github.com/org/repo',
    })
    // The period remains visible.
    expect(parts[2]).toEqual({ kind: 'text', text: '.' })
  })

  // javascript: and data: must never become anchors.
  it('does not link javascript or data URLs', () => {
    // Both schemes are hostile in an href.
    const parts = splitHttpLinks('javascript:alert(1) data:text/html,hi')
    // The whole string stays one text run.
    expect(parts).toEqual([{ kind: 'text', text: 'javascript:alert(1) data:text/html,hi' }])
    // The plain-text bubble layout stays in use.
    expect(messageHasHttpLink('javascript:alert(1)')).toBe(false)
    expect(messageHasHttpLink('data:text/html,hi')).toBe(false)
  })

  // http stays http. Bare hosts and www hosts open as https without requiring the sender to type it.
  it('links https, www, and bare hosts', () => {
    // Explicit http keeps its scheme.
    expect(splitHttpLinks('http://example.com/a')[0]).toEqual({
      kind: 'link',
      text: 'http://example.com/a',
      href: 'http://example.com/a',
    })
    // The three spellings from the chat all become real links.
    expect(splitHttpLinks('https://www.youtube.com/watch?v=abc')[0]).toEqual({
      kind: 'link',
      text: 'https://www.youtube.com/watch?v=abc',
      href: 'https://www.youtube.com/watch?v=abc',
    })
    expect(splitHttpLinks('watch www.youtube.com later')).toEqual([
      { kind: 'text', text: 'watch ' },
      { kind: 'link', text: 'www.youtube.com', href: 'https://www.youtube.com/' },
      { kind: 'text', text: ' later' },
    ])
    expect(splitHttpLinks('see youtube.com.')).toEqual([
      { kind: 'text', text: 'see ' },
      { kind: 'link', text: 'youtube.com', href: 'https://youtube.com/' },
      { kind: 'text', text: '.' },
    ])
    // An email is not a bare host link.
    expect(messageHasHttpLink('sam@youtube.com')).toBe(false)
  })

  // Words before and after the URL stay text. The URL in the middle is still a link.
  it('links a URL that sits between normal words', () => {
    // One sentence, one address in the middle.
    const parts = splitHttpLinks('watch this https://www.youtube.com/watch?v=abc thanks')
    // Prose, the link, then more prose.
    expect(parts).toEqual([
      { kind: 'text', text: 'watch this ' },
      {
        kind: 'link',
        text: 'https://www.youtube.com/watch?v=abc',
        href: 'https://www.youtube.com/watch?v=abc',
      },
      { kind: 'text', text: ' thanks' },
    ])
    // A bare host in the same position is a link too.
    expect(splitHttpLinks('watch this youtube.com thanks')[1]).toEqual({
      kind: 'link',
      text: 'youtube.com',
      href: 'https://youtube.com/',
    })
  })

  // A host is a link alone, at the start, or at the end, for .com, .in, www, and https.
  it('links a host in every position and with other public endings', () => {
    // Alone.
    expect(splitHttpLinks('example.in')).toEqual([
      { kind: 'link', text: 'example.in', href: 'https://example.in/' },
    ])
    // Start of the sentence.
    expect(splitHttpLinks('example.in is the site')[0]).toEqual({
      kind: 'link',
      text: 'example.in',
      href: 'https://example.in/',
    })
    // End of the sentence. The period stays outside the link.
    expect(splitHttpLinks('please open example.in.')).toEqual([
      { kind: 'text', text: 'please open ' },
      { kind: 'link', text: 'example.in', href: 'https://example.in/' },
      { kind: 'text', text: '.' },
    ])
    // A second-level Indian host uses the same .in ending.
    expect(splitHttpLinks('see example.co.in today')[1]).toEqual({
      kind: 'link',
      text: 'example.co.in',
      href: 'https://example.co.in/',
    })
    // www and https forms of the same host.
    expect(splitHttpLinks('www.example.in/page')[0]?.href).toBe('https://www.example.in/page')
    expect(splitHttpLinks('https://example.in/page')[0]?.href).toBe('https://example.in/page')
    // A filename is not a website.
    expect(messageHasHttpLink('notes.pdf')).toBe(false)
    expect(messageHasHttpLink('U.S. news')).toBe(false)
  })
})
