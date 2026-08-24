"""Render untrusted text without allowing it to control the terminal."""

from __future__ import annotations

import unicodedata

from mini_agent.redaction import redact_text


def terminal_safe_text(text: str) -> str:
    """Make terminal control and Unicode formatting characters visible.

    Newlines remain newlines so multi-line output is readable. Every other
    C0 control, DEL, C1 control, and Unicode format character is escaped
    instead of emitted to the terminal.
    """

    rendered: list[str] = []
    for character in text:
        code_point = ord(character)
        if character == "\n":
            rendered.append(character)
        elif code_point <= 0x1F or code_point == 0x7F:
            rendered.append(f"\\x{code_point:02X}")
        elif 0x80 <= code_point <= 0x9F or unicodedata.category(character) == "Cf":
            rendered.append(f"\\u{code_point:04X}")
        else:
            rendered.append(character)
    return "".join(rendered)


def safe_terminal_text(text: str) -> str:
    """Redact text before rendering it safely in an interactive terminal."""

    return terminal_safe_text(redact_text(text).text)
