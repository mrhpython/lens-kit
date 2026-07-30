"""Claim Extraction Pre-Layer.

Extracts structured claims from text before lens validation runs.
Each claim becomes a separate validation target, improving lens accuracy
on short text and isolated claims. Domain-agnostic — no profile vocabulary.
"""
import json
import re
from typing import Dict, List, Optional

import dspy

from .signatures import ClaimExtraction


class ClaimExtractor:
    """Wraps ClaimExtraction signature with parsing and error handling."""

    def __init__(self):
        self.predictor = dspy.Predict(ClaimExtraction)

    def extract(self, text: str) -> List[Dict]:
        """Extract claims from text. Returns list of claim dicts.

        Each claim dict has:
            - claim: str -- the extracted claim text
            - type: str -- quantitative|attributive|predictive|comparative
            - location: str -- where in the text (e.g., "sentence 1")

        Returns empty list on parse failure or no claims found.
        """
        if not text or len(text.strip()) < 10:
            return []

        try:
            result = self.predictor(text=text)
            return self._parse_claims(result.claims)
        except Exception:
            return []

    def _parse_claims(self, raw: str) -> List[Dict]:
        """Parse LLM output into structured claims list."""
        if not raw:
            return []

        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON array in the response
            match = re.search(r'\[.*\]', cleaned, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    return []
            else:
                return []

        if not isinstance(parsed, list):
            return []

        # Validate and normalize each claim
        valid_types = {"quantitative", "attributive", "predictive", "comparative"}
        claims = []
        for item in parsed:
            if not isinstance(item, dict) or "claim" not in item:
                continue
            claim = {
                "claim": str(item["claim"]),
                "type": item.get("type", "quantitative") if item.get("type") in valid_types else "quantitative",
                "location": str(item.get("location", "unknown")),
            }
            claims.append(claim)

        return claims

    def route_claims(self, claims: List[Dict]) -> Dict[str, List[Dict]]:
        """Route extracted claims to appropriate lenses.

        Returns dict with keys: truth, extrapolation, consistency
        Each value is a list of claims relevant to that lens.
        """
        truth_claims = [c for c in claims if c["type"] in ("quantitative", "attributive", "comparative")]
        extrap_claims = [c for c in claims if c["type"] == "predictive"]
        consistency_claims = [c for c in claims if c["type"] == "quantitative"]

        return {
            "truth": truth_claims,
            "extrapolation": extrap_claims,
            "consistency": consistency_claims,
        }

    def format_for_lens(self, claims: List[Dict], lens_name: str) -> str:
        """Format claims as context string to inject into a lens's domain field.

        Returns formatted string to append to the lens's domain/context field.
        Empty string if no claims.
        """
        if not claims:
            return ""

        header = {
            "truth": "EXTRACTED CLAIMS TO VALIDATE (check each for source/evidence):",
            "extrapolation": "EXTRACTED PREDICTIONS TO VALIDATE (check each for confidence/qualification):",
            "consistency": "EXTRACTED NUMBERS TO CROSS-CHECK (verify consistency across claims):",
        }.get(lens_name, "EXTRACTED CLAIMS:")

        lines = [f"\n\n{header}"]
        for c in claims:
            lines.append(f"- [{c['type'].upper()}] {c['claim']} (at: {c['location']})")

        return "\n".join(lines)


def extract_and_route(text: str, extractor: Optional[ClaimExtractor] = None) -> Dict[str, str]:
    """One-call convenience: extract claims, route, and format for each lens."""
    if extractor is None:
        extractor = ClaimExtractor()

    claims = extractor.extract(text)
    if not claims:
        return {"truth": "", "extrapolation": "", "consistency": ""}

    routed = extractor.route_claims(claims)

    return {
        lens: extractor.format_for_lens(routed[lens], lens)
        for lens in ("truth", "extrapolation", "consistency")
    }
