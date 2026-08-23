// Exercise the TypeScript URL-feature port against the Python module's cases.

// Import the browser port of url_features.py.
import { extractMessageUrlFeatures, extractUrls, URL_FEATURE_NAMES } from './urlFeatures'
import { describe, expect, it } from 'vitest'

describe('urlFeatures', () => {
  it('returns a zero vector for a lunch DM with no URL', () => {
    // Score a short legitimate chat line that contains no URL.
    const features = extractMessageUrlFeatures('lunch tomorrow')
    // No URL should be detected.
    expect(features.has_url).toBe(0)
    // URL count should be zero when has_url is zero.
    expect(features.url_count).toBe(0)
    // The vector width must stay 20.
    expect(URL_FEATURE_NAMES).toHaveLength(20)
  })

  it('flags an IP login URL without fetching it', () => {
    // Use documentation-range 192.0.2.1 so the example is not a real host.
    const features = extractMessageUrlFeatures('verify account https://192.0.2.1/login')
    // The URL must be detected.
    expect(features.has_url).toBe(1)
    // The host is an IPv4 literal.
    expect(features.host_is_ip).toBe(1)
    // The path contains a credential-harvesting keyword.
    expect(features.path_has_login_verify_update_password_keywords).toBe(1)
    // HTTPS is used (scheme only; no live fetch).
    expect(features.uses_https).toBe(1)
  })

  it('treats docs.google.com as milder than an IP /verify URL', () => {
    // Ordinary collaborative-doc URL of the kind a legitimate DM shares.
    const hamLike = extractMessageUrlFeatures(
      'see the doc https://docs.google.com/document/d/abc123',
    )
    // Phishing-like IP URL with a verify path.
    const phishLike = extractMessageUrlFeatures('pls check https://192.0.2.1/verify')
    // Both messages contain a URL.
    expect(hamLike.has_url).toBe(1)
    expect(phishLike.has_url).toBe(1)
    // The docs host is not an IP literal.
    expect(hamLike.host_is_ip).toBe(0)
    // The phishing host is an IP literal.
    expect(phishLike.host_is_ip).toBe(1)
    // The docs path should not match login/verify/update/password keywords.
    expect(hamLike.path_has_login_verify_update_password_keywords).toBe(0)
    // The phishing path should match the verify keyword.
    expect(phishLike.path_has_login_verify_update_password_keywords).toBe(1)
  })

  it('flags bit.ly from the frozen shortener list without resolving it', () => {
    // A shortener URL; the extractor must not HTTP-fetch the destination.
    const features = extractMessageUrlFeatures('pls check https://bit.ly/abc123')
    // The frozen list should recognize bit.ly.
    expect(features.is_known_shortener).toBe(1)
    // The host is not an IP.
    expect(features.host_is_ip).toBe(0)
  })

  it('reads href attributes from leftover HTML', () => {
    // Mimic an email body that still contains an anchor tag.
    const urls = extractUrls('<p>Click <a href="https://evil.example/login">here</a> please</p>')
    // The href target must be present even if visible text is only "here".
    expect(urls).toContain('https://evil.example/login')
  })

  it('sets the punycode flag on an xn-- host', () => {
    // xn-- is the ACE prefix for internationalized domain labels.
    const features = extractMessageUrlFeatures('open https://xn--80ak6aa92e.com/login')
    // The punycode flag must be set.
    expect(features.punycode_xn).toBe(1)
  })
})
