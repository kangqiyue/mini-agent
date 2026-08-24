"""Runtime-only synthetic credential values for redaction behavior tests."""

from __future__ import annotations


def synthetic_stripe_access_token(label: str) -> str:
    """Build a credential-shaped test value without publishing one in source."""

    return "sk_" + "live_" + f"SYNTHETIC{label}VALUE12345"


def synthetic_aws_access_key_id() -> str:
    """Build an AWS-shaped test value without publishing one in source."""

    return "A" + "K" + "IA" + "0" * 16
