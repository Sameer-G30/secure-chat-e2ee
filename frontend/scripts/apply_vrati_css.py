#!/usr/bin/env python3
"""Adapt the frontend/vrati ink-and-wax stylesheet onto html[data-theme]."""

# Read regular expressions so the token blocks can be swapped in place.
import re
# Read the process helpers used to locate the downloaded source stylesheet.
import sys

# Point at the vrati CSS curl already saved under /tmp.
SOURCE = "/tmp/vrati-index.css"
# Write the adapted sheet into the Vite frontend the app actually loads.
DEST = "/home/sam/projects/Minor Project-II/frontend/src/index.css"

# Extra rules this stack needs that vrati never shipped (ML, tests, aliases).
EXTRAS = r"""

/* ==========================================================================
   Stack-specific extras (not in frontend/vrati)
   Light is the signed-out / default palette. Dark is html[data-theme='dark'].
   ========================================================================== */

/* Hide a label visually while keeping it available to assistive technology. */
.visually-hidden {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
}

/* Let the sequential ORT Web table scroll even though the chat shell is locked. */
html:has(.load-check-screen),
html:has(.load-check-screen) body,
html:has(.load-check-screen) #root {
    height: auto;
    overflow: auto;
}

/* Style the measurement page with the same paper tokens as the rest of the app. */
.load-check-screen {
    max-width: 1100px;
    margin: 0 auto;
    padding: 24px;
    background: var(--bg-primary);
    min-height: 100svh;
    color: var(--text-primary);
}

/* Draw the load-check table on the card surface. */
.load-check-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    background: var(--bg-secondary);
    color: var(--text-primary);
}

/* Pad and edge every measurement cell. */
.load-check-table th,
.load-check-table td {
    border: 1px solid var(--border-color);
    padding: 8px;
    text-align: left;
    vertical-align: top;
}

/* Space the JSON dump label from the table. */
.load-check-json-label {
    display: block;
    margin-top: 16px;
    font-size: 13px;
}

/* Show the raw JSON log in the mono face. */
.load-check-json {
    width: 100%;
    min-height: 200px;
    font-family: var(--font-mono);
    font-size: 12px;
    background: var(--bg-tertiary);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
}

/* Alias legacy primary-button class names still used in modal actions. */
.primary-button {
    padding: 10px 22px;
    background: var(--accent);
    color: #fdf6f0;
    border: none;
    border-radius: var(--radius-full);
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    font-family: var(--font-body);
    transition: background var(--transition);
}

/* Darken the alias on hover, matching .modal-btn-confirm. */
.primary-button:hover:not(:disabled) {
    background: var(--accent-hover);
}

/* Dim a disabled primary action. */
.primary-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

/* Style in-page mode switches as accent links, not fake anchors. */
.text-button {
    background: none;
    border: none;
    padding: 0;
    color: var(--accent);
    cursor: pointer;
    font: inherit;
    font-weight: 500;
    text-decoration: none;
}

/* Soften the link-button on hover. */
.text-button:hover {
    opacity: 0.75;
}

/* Keep the auth footer switch looking like vrati's <a> treatment. */
.form-footer .text-button {
    color: var(--accent);
    font-size: 14px;
}

/* Stack the live password meter under the registration field. */
.password-strength {
    display: grid;
    gap: 6px;
    margin-top: 8px;
}

/* Draw the empty meter track. */
.password-strength-track {
    height: 6px;
    border-radius: var(--radius-full);
    background: var(--border-color);
    overflow: hidden;
}

/* Fill the meter with a warning amber by default. */
.password-strength-fill {
    height: 100%;
    border-radius: var(--radius-full);
    background: #b45309;
}

/* Shift mid scores toward seal gold. */
.password-strength-fill-3,
.password-strength-fill-4 {
    background: var(--seal);
}

/* Paint a passing score green. */
.password-strength-fill-5 {
    background: #15803d;
}

/* Caption the meter with supporting copy. */
.password-strength-label {
    margin: 0;
    color: var(--text-secondary);
    font-size: 12px;
}

/* Position the show/hide control over the password field. */
.password-toggle-wrap {
    position: relative;
}

/* Keep typed characters clear of the toggle. */
.password-toggle-wrap input {
    padding-right: 52px;
}

/* Place the show/hide control on the field's trailing edge. */
.password-toggle {
    position: absolute;
    right: 12px;
    top: 50%;
    transform: translateY(-50%);
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    font-size: 13px;
    font-family: var(--font-mono);
    padding: 4px;
}

/* Announce auth and chat outcomes without looking like a bubble. */
.auth-feedback {
    margin: 12px 0 0;
    padding: 10px 12px;
    border-radius: var(--radius-sm);
    font-size: 13px;
    line-height: 1.45;
}

/* Tint success copy green. */
.auth-feedback-success {
    color: #15803d;
    background: rgba(21, 128, 61, 0.12);
}

/* Tint error copy with the wax-seal danger token. */
.auth-feedback-error {
    color: var(--danger);
    background: var(--danger-light);
}

/* Keep submitting/idle notes in supporting copy. */
.auth-feedback-submitting,
.auth-feedback-idle {
    color: var(--text-secondary);
}

/* Non-blocking scam warning; never hides or deletes the plaintext. */
.scam-banner {
    margin: 0 0 8px;
    padding: 6px 8px;
    border-radius: var(--radius-sm);
    font-size: 12px;
    font-weight: 600;
    line-height: 1.3;
    color: #92400e;
    background: var(--seal-light);
}

/* Keep the banner readable on an oxblood sent bubble. */
.message.sent .scam-banner {
    color: #fff7ed;
    background: rgba(0, 0, 0, 0.22);
}

/* Show authentication failure as a distinct state, never as garbled plaintext. */
.message.failed {
    color: var(--danger);
    background: var(--danger-light);
    border: 1px solid rgba(176, 65, 62, 0.25);
    align-self: flex-start;
}

/* Reset the bubble-as-button so tests can click plaintext without extra chrome. */
button.message {
    font: inherit;
    text-align: left;
    appearance: none;
    -webkit-appearance: none;
    border: none;
    display: block;
}

/* Keep a failed bubble's button chrome aligned with .message.failed. */
button.message.failed {
    border: 1px solid rgba(176, 65, 62, 0.25);
}

/* Strip list markers from the transcript and the address book. */
ul.messages,
ul.contacts-list,
ul.search-results-list {
    list-style: none;
    margin: 0;
    padding: 0;
}

/* Let .messages keep vrati's flex transcript layout when it is a ul. */
ul.messages {
    flex: 1;
    padding: 20px 24px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 6px;
    background: var(--bg-primary);
}

/* Stretch each contact row so Remove can sit beside the handle. */
.contact-row {
    display: flex;
    align-items: stretch;
}

/* Make the contact hit-target a real button without losing vrati padding. */
button.contact-item {
    flex: 1;
    background: none;
    border: none;
    border-left: 2px solid transparent;
    font: inherit;
    color: inherit;
    text-align: left;
}

/* Keep the active contact's oxblood rail when the row is a button. */
button.contact-item.active {
    border-left-color: var(--accent);
}

/* Quiet remove action on the contact row. */
.contact-remove {
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    font-size: 11px;
    font-family: var(--font-mono);
    padding: 0 16px 0 0;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* Emphasize remove on hover. */
.contact-remove:hover {
    color: var(--danger);
}

/* Pin on-device model opt-ins to the sidebar footer so tests always find them. */
.sidebar-footer {
    padding: 12px 16px 16px;
    border-top: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    gap: 8px;
    flex-shrink: 0;
}

/* Stack DistilBERT and Word BiLSTM checkboxes without crowding the header. */
.chat-model-toggles {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 6px;
}

/* Keep the on-device model toggles readable on the sidebar. */
.chat-model-toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: var(--text-secondary);
    font-family: var(--font-body);
}

/* Announce connection state without looking like an error. */
.chat-status {
    margin: 0;
    padding: 8px 24px 0;
    color: var(--text-secondary);
    font-size: 12px;
    font-family: var(--font-mono);
}

/* Keep typing notices next to the transcript. */
.chat-typing {
    margin: 0;
    padding: 4px 24px 0;
    color: var(--seal);
    font-size: 12px;
    font-family: var(--font-mono);
}

/* Hide the product heading that tests query without duplicating the peer name. */
.chat-product-heading {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
}

/* Group modal actions on one row. */
.chat-modal-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 16px;
}

/* Stack per-bubble actions vertically inside the dialog. */
.chat-modal-actions-stack {
    flex-direction: column;
    align-items: stretch;
}

/* Stretch stacked message-action aliases to the dialog width. */
.chat-modal-actions-stack .text-button,
.chat-modal-actions-stack .message-action-btn {
    width: 100%;
    text-align: left;
}

/* Space the report and edit textareas like vrati's report field. */
.chat-report-form textarea,
.chat-edit-form textarea {
    width: 100%;
    padding: 12px 16px;
    background: var(--bg-tertiary);
    border: 1.5px solid var(--border-color);
    border-radius: var(--radius-sm);
    font-size: 14px;
    color: var(--text-primary);
    outline: none;
    font-family: var(--font-body);
    resize: vertical;
    min-height: 88px;
}

/* Glow the report/edit field on focus. */
.chat-report-form textarea:focus,
.chat-edit-form textarea:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-glow);
}

/* Keep settings rows as buttons so they stay keyboard-accessible. */
button.settings-item {
    width: 100%;
    background: none;
    border: none;
    font: inherit;
    color: inherit;
    text-align: left;
}

/* Tint the logout row with the danger token. */
.settings-item-logout {
    color: var(--danger);
}

/* Keep theme radios visually hidden inside the pretty option cards. */
.theme-option input[type="radio"] {
    position: absolute;
    opacity: 0;
    width: 1px;
    height: 1px;
}

/* Position each theme card so the hidden radio still belongs to the label. */
label.theme-option {
    display: block;
    position: relative;
}

/* Dim a disabled circular send control. */
.btn-send:disabled {
    opacity: 0.45;
    cursor: not-allowed;
}

/* Undo vrati's "last dropdown item is always danger" so Clear is not red. */
.dropdown-item:last-child {
    color: var(--text-primary);
}

/* Undo the matching last-child icon tint. */
.dropdown-item:last-child svg {
    color: var(--text-secondary);
}

/* Mark block/report as destructive actions explicitly. */
.dropdown-item-danger,
.dropdown-item-danger svg {
    color: var(--danger);
}

/* Disable more-menu items the same way vrati greys empty actions. */
.dropdown-item:disabled {
    opacity: 0.4;
    cursor: not-allowed;
}

/* Let the in-chat search Close control keep an accessible name. */
.search-close-btn {
    font-size: 12px;
    font-family: var(--font-mono);
    gap: 4px;
}

/* Keep search-hit buttons unstyled besides vrati's result row. */
button.search-result-item {
    width: 100%;
    background: none;
    border: none;
    font: inherit;
    color: inherit;
    text-align: left;
    cursor: pointer;
}

/* Empty-search copy under the add-contact field. */
.chat-search-empty {
    margin: 0;
    padding: 8px 20px;
    font-size: 13px;
    color: var(--text-muted);
}

/* Caption unimplemented settings pages. */
.settings-stub {
    padding: 20px 26px;
    color: var(--text-secondary);
    font-size: 14px;
    line-height: 1.6;
}

/* Keep code snippets in modal copy on the mono face. */
.modal-body code {
    font-family: var(--font-mono);
    font-size: 12px;
}
"""


def main() -> int:
    """Rewrite vrati tokens onto html[data-theme] and append extras."""
    # Load the downloaded vrati stylesheet as one string.
    css = open(SOURCE, encoding="utf-8").read()
    # Capture the original :root block (dark tokens plus shared radii/fonts).
    root_match = re.search(r":root \{.*?\n\}", css, flags=re.S)
    # Capture the original light-mode override block.
    light_match = re.search(r"body\.light-mode \{.*?\n\}", css, flags=re.S)
    # Abort if the expected token blocks are missing.
    if root_match is None or light_match is None:
        # Tell the caller which block was absent.
        print("could not find :root or body.light-mode", file=sys.stderr)
        # Signal failure to the shell.
        return 1
    # Keep the original dark :root text for later dark-theme overrides.
    root_block = root_match.group(0)
    # Keep the original light override text so its values can move into :root.
    light_block = light_match.group(0)
    # Parse every CSS custom property from the light-mode block.
    light_vars = dict(re.findall(r"(--[\w-]+):\s*([^;]+);", light_block))
    # Parse every CSS custom property from the original (dark) :root block.
    dark_vars = dict(re.findall(r"(--[\w-]+):\s*([^;]+);", root_block))
    # Start from the full :root block so radii, fonts, and unused tokens remain.
    new_root = root_block
    # Overwrite each color token in :root with the light-mode paper value.
    for name, value in light_vars.items():
        # Replace only the first occurrence of this custom property.
        new_root = re.sub(
            rf"({re.escape(name)}:\s*)[^;]+;",
            rf"\g<1>{value.strip()};",
            new_root,
            count=1,
        )
    # Insert an explicit light color-scheme for native form controls.
    new_root = new_root.replace(":root {", ":root {\n    color-scheme: light;", 1)
    # Rebuild a dark-theme override that restores only the swapped color tokens.
    dark_lines = ["html[data-theme='dark'] {", "    color-scheme: dark;"]
    # Emit each light-overridden token with its original dark value.
    for name in light_vars:
        # Skip tokens the dark :root never defined.
        if name not in dark_vars:
            # Leave unknown names out of the dark override.
            continue
        # Write the dark value back for this token.
        dark_lines.append(f"    {name}: {dark_vars[name].strip()};")
    # Close the dark-theme override block.
    dark_lines.append("}")
    # Join the dark override into one stylesheet fragment.
    dark_block = "\n".join(dark_lines)
    # Splice the new light :root and the dark override into the sheet.
    css = css[: root_match.start()] + new_root + "\n\n" + dark_block + css[root_match.end() :]
    # Drop the original body.light-mode block so it cannot fight html[data-theme].
    css = re.sub(r"\nbody\.light-mode \{.*?\n\}", "\n", css, count=1, flags=re.S)
    # Point the brand-icon fill at the oxblood accent instead of leftover indigo.
    css = css.replace("fill: #4f46e5;", "fill: var(--accent);")
    # Append the extras this stack still needs after the vrati rules.
    css = css.rstrip() + "\n" + EXTRAS
    # Write the finished stylesheet into the Vite source tree.
    open(DEST, "w", encoding="utf-8").write(css)
    # Print a short confirmation so the shell log is easy to scan.
    print(f"wrote {DEST} ({len(css)} bytes)")
    # Signal success.
    return 0


# Run the adapter when this file is executed as a script.
if __name__ == "__main__":
    # Exit with the adapter's status code.
    raise SystemExit(main())
