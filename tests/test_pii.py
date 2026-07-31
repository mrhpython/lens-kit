"""PII module tests — deterministic, no LLM calls.

These are the structured patterns the internal eval (2026-03) showed an
LLM-only Rights lens missing ~50% of the time. Regex catches them all.
Every key/card/number below is a fake fixture.
"""

from lens_kit.pii import scan, scrub


# ═══════════════════════════════════════════════════════════════════
# EMAIL / PHONE (warning severity — never halt, never fail the gate)
# ═══════════════════════════════════════════════════════════════════

class TestWarningSeverity:

    def test_catches_email(self):
        result = scan("Contact john.smith@company.com for details")
        assert result.has_pii
        assert any(m.pii_type == "email" for m in result.matches)
        assert not result.halt

    def test_scrubs_email(self):
        clean = scrub("Contact john.smith@company.com for details")
        assert "john.smith@company.com" not in clean
        assert "[REDACTED]" in clean

    def test_catches_uk_mobile(self):
        result = scan("Call Sarah on 07912345678")
        assert result.has_pii
        assert any(m.pii_type == "phone_uk" for m in result.matches)
        assert not result.halt

    def test_us_intl_prefix_included_in_match(self):
        """Leading \\b before +1 never matches (\\b needs a word char);
        (?<!\\w) keeps '+1 (' inside the redaction."""
        clean = scrub("Reach me on +1 (555) 014-2368 today")
        assert "+1 ([REDACTED]" not in clean
        assert "555" not in clean
        assert "[REDACTED]" in clean

    def test_no_flag_redacted(self):
        result = scan("Contact [REDACTED] for details")
        assert not any(m.pii_type == "email" for m in result.matches)


# ═══════════════════════════════════════════════════════════════════
# CRITICAL + HALT TYPES
# ═══════════════════════════════════════════════════════════════════

class TestCriticalHalts:

    def test_credit_card_halts(self):
        result = scan("Card on file: 4111 1111 1111 1111")
        assert result.halt
        assert any(m.pii_type == "credit_card" for m in result.matches)

    def test_ni_number_halts(self):
        result = scan("His NI is QQ 12 34 56 C")
        assert result.halt
        assert any(m.pii_type == "ni_number" for m in result.matches)

    def test_ssn_halts(self):
        result = scan("SSN 078-05-1120 on the form")
        assert result.halt
        assert any(m.pii_type == "ssn" for m in result.matches)

    def test_credential_assignment_halts(self):
        result = scan("DB_PASSWORD=hunter2hunter2")
        assert result.halt
        assert any(m.pii_type == "credential" for m in result.matches)

    def test_connection_string_detected(self):
        result = scan("postgres://admin:s3cretpass@db.internal:5432/prod")
        assert any(m.pii_type == "connection_string" for m in result.matches)

    def test_destructive_cmd_critical_but_no_halt(self):
        """Content hazard, not a secret — flags critical, does not halt."""
        result = scan("then run rm -rf / to clean up")
        assert any(m.pii_type == "destructive_cmd" for m in result.matches)
        assert not result.halt


class TestAPIKeyDetection:

    def test_catches_sk_dash_key(self):
        result = scan("KEY=sk-ant-api03-xK9mN2pL5qR8vT1wY4zA7bC0dE3fG6hJ")
        assert result.halt
        assert any(m.pii_type == "api_key" for m in result.matches)

    def test_catches_stripe_style_live_key(self):
        """The gap that motivated this module going public: sk_live_
        (underscore form) slipped past an sk- (hyphen) pattern.

        The fixture key is assembled at runtime so secret scanners
        (GitHub push protection, gitleaks) never see a contiguous
        key-shaped literal in the source."""
        fake_key = "sk_" + "live_" + "deadbeef" * 6
        result = scan(f"old key {fake_key} embedded in the config")
        assert result.halt
        assert any(m.pii_type == "api_key" for m in result.matches)

    def test_catches_github_pat(self):
        result = scan("token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn")
        assert any(m.pii_type == "api_key" for m in result.matches)

    def test_catches_aws_key(self):
        result = scan("AWS_KEY=AKIAIOSFODNN7EXAMPLE")
        assert any(m.pii_type == "api_key" for m in result.matches)

    def test_scrubs_api_key(self):
        clean = scrub("KEY=sk-ant-api03-xK9mN2pL5qR8vT1wY4zA7bC0dE3fG6hJ")
        assert "sk-ant" not in clean
        assert "[REDACTED]" in clean


# ═══════════════════════════════════════════════════════════════════
# CLEAN TEXT
# ═══════════════════════════════════════════════════════════════════

class TestCleanText:

    def test_plain_prose_is_clean(self):
        result = scan("The quarterly report shows steady progress across the "
                      "three workstreams, with delivery expected in Q3.")
        assert not result.has_pii
        assert result.scrubbed_text.startswith("The quarterly report")

    def test_version_numbers_not_flagged(self):
        result = scan("Upgrade from version 3.5.27 to 4.0.1 is recommended")
        assert not result.has_pii

    def test_ordinary_numbers_not_flagged(self):
        result = scan("Revenue grew 42% across 17 accounts in 3 regions")
        assert not result.has_pii
