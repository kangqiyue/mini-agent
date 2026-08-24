from mini_agent.terminal_safety import safe_terminal_text, terminal_safe_text


def test_terminal_safe_text_escapes_osc_sequence() -> None:
    text = "before\x1b]52;c;clipboard\x1b\\after"

    assert terminal_safe_text(text) == "before\\x1B]52;c;clipboard\\x1B\\after"


def test_terminal_safe_text_escapes_c1_and_bidi_controls() -> None:
    text = "left\u009Bright\u202eend"

    assert terminal_safe_text(text) == "left\\u009Bright\\u202Eend"


def test_terminal_safe_text_preserves_newlines() -> None:
    assert terminal_safe_text("first\nsecond") == "first\nsecond"


def test_terminal_display_redacts_before_escaping_controls() -> None:
    unsafe_text = "api_key=sk_live_0000000000000000 \x1b]52;c;clipboard\x1b\\"

    displayed = safe_terminal_text(unsafe_text)

    assert "sk_live_0000000000000000" not in displayed
    assert "[REDACTED]" in displayed
    assert "\\x1B]52;c;clipboard\\x1B\\" in displayed
