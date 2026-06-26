from search_assistant.verification.policy import (
    extract_key_claims,
    requires_calibration,
    requires_verification,
)


def test_current_version_claim_requires_verification():
    assert requires_verification("FastAPI 0.138.1 is the latest version.", "research")


def test_hard_and_high_stakes_require_calibration():
    assert requires_calibration("hard", has_unverified_claims=False)
    assert requires_calibration("high_stakes", has_unverified_claims=False)
    assert requires_calibration("simple", has_unverified_claims=True)


def test_extract_key_claims_keeps_numbered_claims():
    claims = extract_key_claims("FastAPI 0.138.1 was released in 2026. Use official docs.")

    assert "FastAPI 0.138.1 was released in 2026" in claims
