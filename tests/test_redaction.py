import json

import pytest
from pydantic import ValidationError

from mini_agent.redaction import redact_json_text, redact_text
from mini_agent.redaction_types import RedactionKind, RedactionSummary

_STRIPE_LIVE_KEY = "sk_live_" + "0" * 24
_STRIPE_TEST_KEY = "sk_test_" + "0" * 24
_STRIPE_RESTRICTED_LIVE_KEY = "rk_live_" + "0" * 24
_STRIPE_RESTRICTED_TEST_KEY = "rk_test_" + "0" * 24
_STRIPE_WEBHOOK_SECRET = "whsec_" + "0" * 24
_AWS_ACCESS_KEY_ID = "AKIA" + "0" * 16
_COMPACT_SENSITIVE_FIELD_NAMES = (
    "apikey",
    "accesstoken",
    "secretaccesskey",
    "refreshtoken",
    "privatekey",
    "connectionstring",
    "databaseurl",
    "secretkey",
)
_PARTIAL_SECRET_TAILS = (
    "sk_live_0",
    "sk_test_0",
    "rk_live_0",
    "rk_test_0",
    "whsec_0",
    "AKIA0",
    "ASIA0",
    "sk-0",
    "ghp_0",
    "github_pat_0",
    "xoxb",
)


@pytest.mark.parametrize(
    ("unsafe_text", "secret"),
    [
        ("Authorization: Bearer test-value", "test-value"),
        ("Authorization: Basic test-value", "test-value"),
        ("Authorization=Bearer test-equals-value", "test-equals-value"),
        ("Authorization = Basic test-equals-basic", "test-equals-basic"),
        ("api_" + "key=test-value", "test-value"),
        ("OPENAI_API_KEY=test-value", "test-value"),
        ("ANTHROPIC_API_KEY: test-value", "test-value"),
        ("AWS_ACCESS_KEY_ID=test-aws-id", "test-aws-id"),
        ("AWS_SECRET_ACCESS_KEY=test-aws-secret", "test-aws-secret"),
        ("STRIPE_SECRET_KEY=test-stripe-secret", "test-stripe-secret"),
        ("pass" + "word: test-value", "test-value"),
        ("Cookie: session=test-value; theme=dark", "test-value"),
        ("https://test-user:test-http-password@example.invalid/path", "test-http-password"),
        ("postgresql://user:test-value@db.example/app", "test-value"),
        ("Stripe live key " + _STRIPE_LIVE_KEY, _STRIPE_LIVE_KEY),
        ("Stripe test key " + _STRIPE_TEST_KEY, _STRIPE_TEST_KEY),
        (
            "Stripe restricted live key " + _STRIPE_RESTRICTED_LIVE_KEY,
            _STRIPE_RESTRICTED_LIVE_KEY,
        ),
        (
            "Stripe restricted test key " + _STRIPE_RESTRICTED_TEST_KEY,
            _STRIPE_RESTRICTED_TEST_KEY,
        ),
        ("Stripe webhook secret " + _STRIPE_WEBHOOK_SECRET, _STRIPE_WEBHOOK_SECRET),
        ("AWS access key " + _AWS_ACCESS_KEY_ID, _AWS_ACCESS_KEY_ID),
        ("token " + "sk-" + "x" * 16, "sk-" + "x" * 16),
        (
            "-" * 5
            + "BEGIN PRIVATE KEY"
            + "-" * 5
            + "\ntest-value\n"
            + "-" * 5
            + "END PRIVATE KEY"
            + "-" * 5,
            "test-value",
        ),
    ],
)
def test_redact_text_removes_supported_credentials(unsafe_text: str, secret: str) -> None:
    result = redact_text(unsafe_text)

    assert secret not in result.text
    assert "[REDACTED]" in result.text
    assert result.match_count >= 1


def test_redact_text_leaves_regular_text_unchanged() -> None:
    result = redact_text("Explain append-only event logs")

    assert result.text == "Explain append-only event logs"
    assert result.match_count == 0
    assert result.matched_kinds == frozenset()


@pytest.mark.parametrize(
    ("match_count", "kinds"),
    [
        (0, (RedactionKind.NAMED_SECRET,)),
        (1, ()),
    ],
)
def test_redaction_summary_rejects_ambiguous_count_and_kinds(
    match_count: int,
    kinds: tuple[RedactionKind, ...],
) -> None:
    with pytest.raises(ValidationError, match="match count must be zero"):
        RedactionSummary(match_count=match_count, kinds=kinds)


def test_redact_text_does_not_redact_similarly_named_regular_fields() -> None:
    regular_text = (
        "token_count=12 retry_token_count=3 api_key_hint=read "
        "access_key_count=2 secret_key_name=primary "
        "rk_" + "live_identifier whsec_" + "docs https://example.invalid/docs"
    )
    result = redact_text(regular_text)

    assert result.text == regular_text
    assert result.match_count == 0


def test_redact_text_redacts_sensitive_http_query_values_without_rewriting_url() -> None:
    unsafe_value = "synthetic-url-secret"
    text = (
        "See https://example.test/v1?token_count=12&api_key="
        f"{unsafe_value}&mode=fast#section=ordinary-fragment"
    )

    result = redact_text(text)

    assert unsafe_value not in result.text
    assert result.text == (
        "See https://example.test/v1?token_count=12&api_key=[REDACTED]"
        "&mode=fast#section=ordinary-fragment"
    )
    assert result.match_count == 1
    assert result.matched_kinds == frozenset((RedactionKind.NAMED_SECRET,))


def test_redact_text_redacts_percent_encoded_sensitive_query_key_and_value() -> None:
    encoded_value = "synthetic%2Durl%2Dsecret"
    result = redact_text(
        "https://example.test/v1?api%5Fkey=" + encoded_value + "&token_count=12"
    )

    assert encoded_value not in result.text
    assert result.text == "https://example.test/v1?api%5Fkey=[REDACTED]&token_count=12"
    assert result.match_count == 1


def test_redact_text_redacts_sensitive_http_fragment_parameters() -> None:
    unsafe_value = "synthetic-oauth-fragment-secret"
    result = redact_text(
        "https://example.test/callback#state=ordinary&access_token=" + unsafe_value
    )

    assert unsafe_value not in result.text
    assert result.text == "https://example.test/callback#state=ordinary&access_token=[REDACTED]"
    assert result.match_count == 1


def test_redact_json_text_redacts_sensitive_http_query_value() -> None:
    unsafe_value = "synthetic-json-url-secret"
    result = redact_json_text(
        json.dumps({"content": f"https://example.test/v1?api_key={unsafe_value}"})
    )

    assert unsafe_value not in result.text
    assert json.loads(result.text) == {
        "content": "https://example.test/v1?api_key=[REDACTED]"
    }


def test_redact_json_text_redacts_bare_stripe_restricted_and_webhook_secrets() -> None:
    unsafe_json = json.dumps(
        {
            "content": (
                f"live={_STRIPE_RESTRICTED_LIVE_KEY} "
                f"test={_STRIPE_RESTRICTED_TEST_KEY} "
                f"webhook={_STRIPE_WEBHOOK_SECRET}"
            )
        }
    )

    result = redact_json_text(unsafe_json)

    assert json.loads(result.text) == {
        "content": "live=[REDACTED] test=[REDACTED] webhook=[REDACTED]"
    }
    assert result.match_count == 3


def test_redact_text_redacts_secret_value_and_secret_string_named_formats() -> None:
    unsafe_text = "secretValue=synthetic-value secret_string: another-synthetic-value"
    result = redact_text(unsafe_text)

    assert "synthetic-value" not in result.text
    assert "another-synthetic-value" not in result.text
    assert result.text == "secretValue=[REDACTED] secret_string: [REDACTED]"


def test_redact_text_redacts_a_multiline_quoted_named_secret() -> None:
    unsafe_text = 'api_key="synthetic-first-line\nsynthetic-second-line" after'

    result = redact_text(unsafe_text)

    assert "synthetic-first-line" not in result.text
    assert "synthetic-second-line" not in result.text
    assert result.text == "api_key=[REDACTED] after"
    assert result.match_count == 1


def test_redact_text_redacts_common_bare_token_formats() -> None:
    tokens = (
        "AIza" + "A" * 35,
        "hf_" + "B" * 24,
        "glpat-" + "C" * 24,
        f"{'D' * 20}.{'E' * 20}.{'F' * 20}",
    )

    result = redact_text(" ".join(tokens))

    assert all(token not in result.text for token in tokens)
    assert result.text == " ".join("[REDACTED]" for _ in tokens)
    assert result.match_count == len(tokens)


def test_compact_sensitive_field_names_are_redacted_in_text_and_json() -> None:
    unsafe_values = {
        name: f"synthetic-compact-value-{index}"
        for index, name in enumerate(_COMPACT_SENSITIVE_FIELD_NAMES)
    }

    text_result = redact_text(
        " ".join(f"{name}={value}" for name, value in unsafe_values.items())
    )
    json_result = redact_json_text(json.dumps(unsafe_values))

    for value in unsafe_values.values():
        assert value not in text_result.text
        assert value not in json_result.text
    assert text_result.text.count("[REDACTED]") == len(unsafe_values)
    assert json.loads(json_result.text) == {
        name: "[REDACTED]" for name in unsafe_values
    }
    assert text_result.match_count == len(unsafe_values)
    assert json_result.match_count == len(unsafe_values)


def test_compact_sensitive_field_names_do_not_redact_regular_suffixes() -> None:
    regular_text = "apikey_hint=read secretkey_name=primary token_count=12"

    result = redact_text(regular_text)

    assert result.text == regular_text
    assert result.match_count == 0


def test_redact_text_does_not_consume_newlines_before_a_sensitive_assignment() -> None:
    unsafe_text = "stdout:\napikey=synthetic-multiline-value"

    result = redact_text(unsafe_text)

    assert "synthetic-multiline-value" not in result.text
    assert result.text == "stdout:\napikey=[REDACTED]"
    assert result.match_count == 1


@pytest.mark.parametrize("partial_secret", _PARTIAL_SECRET_TAILS)
def test_redact_text_removes_partial_secret_prefix_at_truncated_boundary(
    partial_secret: str,
) -> None:
    unsafe_text = f"command output: {partial_secret}"

    result = redact_text(unsafe_text, truncated_at_end=True)

    assert partial_secret not in result.text
    assert result.text == "command output: [REDACTED]"
    assert result.match_count == 1


@pytest.mark.parametrize("partial_secret", _PARTIAL_SECRET_TAILS)
def test_redact_text_retains_short_secret_prefix_at_natural_text_end(
    partial_secret: str,
) -> None:
    text = f"documented prefix: {partial_secret}"

    result = redact_text(text)

    assert result.text == text
    assert result.match_count == 0


def test_redact_text_retains_bare_documented_secret_prefixes() -> None:
    text = "Use the sk_live_ prefix for production keys."

    result = redact_text(text)

    assert result.text == text
    assert result.match_count == 0


def test_redact_text_removes_unterminated_private_key_at_text_end() -> None:
    unsafe_text = "stdout:\n-----BEGIN PRIVATE KEY-----\npartial-private-material"

    result = redact_text(unsafe_text, truncated_at_end=True)

    assert "partial-private-material" not in result.text
    assert result.text == "stdout:\n[REDACTED]"
    assert result.match_count == 1


def test_redact_json_text_redacts_sensitive_key_suffixes_without_overreaching() -> None:
    unsafe_json = json.dumps(
        {
            "openai_api_key": "test-openai-secret",
            "openaiApiKey": "test-openai-camel-secret",
            "database_url": "postgresql://user:test-db-secret@db.example/app",
            "nested": {"ANTHROPIC_API_KEY": "test-anthropic-secret"},
            "aws": {
                "AWS_ACCESS_KEY_ID": "test-aws-id",
                "AWS_SECRET_ACCESS_KEY": "test-aws-secret",
                "stripeSecretKey": "test-stripe-secret",
                "secretString": "test-aws-secret-string",
            },
            "token_count": 12,
            "note": "api_key is a configuration field",
        }
    )

    result = redact_json_text(unsafe_json)
    safe_json = json.loads(result.text)

    assert "test-openai-secret" not in result.text
    assert "test-openai-camel-secret" not in result.text
    assert "test-db-secret" not in result.text
    assert "test-anthropic-secret" not in result.text
    assert "test-aws-id" not in result.text
    assert "test-aws-secret" not in result.text
    assert "test-stripe-secret" not in result.text
    assert "test-aws-secret-string" not in result.text
    assert safe_json["openai_api_key"] == "[REDACTED]"
    assert safe_json["openaiApiKey"] == "[REDACTED]"
    assert safe_json["database_url"] == "[REDACTED]"
    assert safe_json["nested"]["ANTHROPIC_API_KEY"] == "[REDACTED]"
    assert safe_json["aws"] == {
        "AWS_ACCESS_KEY_ID": "[REDACTED]",
        "AWS_SECRET_ACCESS_KEY": "[REDACTED]",
        "stripeSecretKey": "[REDACTED]",
        "secretString": "[REDACTED]",
    }
    assert safe_json["token_count"] == 12
    assert safe_json["note"] == "api_key is a configuration field"


def test_redact_json_text_redacts_credential_shaped_object_keys_without_collision() -> None:
    first_secret_key = "sk_live_" + "0" * 24
    second_secret_key = "rk_test_" + "0" * 24
    unsafe_json = json.dumps(
        {
            "[REDACTED_KEY_1]": "ordinary field is retained",
            first_secret_key: "first secret-key field",
            second_secret_key: "second secret-key field",
        }
    )

    result = redact_json_text(unsafe_json)

    assert first_secret_key not in result.text
    assert second_secret_key not in result.text
    assert result.match_count == 2
    assert json.loads(result.text) == {
        "[REDACTED_KEY_1]": "ordinary field is retained",
        "[REDACTED_KEY_2]": "second secret-key field",
        "[REDACTED_KEY_3]": "first secret-key field",
    }
    assert redact_json_text(
        json.dumps(
            {
                second_secret_key: "second secret-key field",
                first_secret_key: "first secret-key field",
                "[REDACTED_KEY_1]": "ordinary field is retained",
            }
        )
    ).text == result.text


def test_redact_json_text_falls_back_to_plain_text_for_invalid_json() -> None:
    result = redact_json_text('{"OPENAI_API_KEY":"test-value"')

    assert "test-value" not in result.text
    assert result.text == '{"OPENAI_API_KEY":[REDACTED]'


@pytest.mark.parametrize(
    "text",
    [
        "request header: Bearer ya29.Gl0Bdabc-1234567890_abcdefghij",
        "opaque: Bearer abcdefghijklmnop1234567890qrstuvwxyz",
        "token=Bearer Zm9vYmFyYmF6MTIzNDU2Nzg5MDEyMzQ1",
        "Basic Z2VuZXJpYzpwYXNzd29yZDEyMzQ1Njc4OQ==",
    ],
)
def test_redact_text_redacts_bare_bearer_or_basic_token(text: str) -> None:
    # A bearer/basic credential echoed in prose without an ``Authorization:``
    # prefix must still be redacted; the 16-char token minimum keeps ordinary
    # phrases intact.
    result = redact_text(text)

    assert "Bearer" not in result.text or "[REDACTED]" in result.text
    assert "ya29" not in result.text
    assert "abcdefghijklmnop1234567890qrstuvwxyz" not in result.text
    assert "Zm9vYmFy" not in result.text
    assert "Z2VuZXJpYzpwYXNz" not in result.text
    assert result.match_count >= 1


@pytest.mark.parametrize(
    "text",
    [
        "bearer of bad news",
        "basic authentication",
        "basic knowledge required",
        "basic internationalization",
        "basic cross-functional",
        "basic well-established",
    ],
)
def test_redact_text_leaves_bearer_basic_prose_intact(text: str) -> None:
    assert redact_text(text).match_count == 0
