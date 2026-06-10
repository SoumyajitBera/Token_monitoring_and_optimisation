"""Generic evidence-consistency checks for structured agent outputs.

This validator is intentionally domain-light. It does not decide whether the
LLM is "right". It catches obvious contradictions in normalized outputs:
- evidence marked missing but decision is extremely positive;
- risks and reasons containing near-identical claims;
- high scores with many missing-evidence items;
- empty reasons with positive decisions.

No external dependencies are used.
"""

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Set


@dataclass
class EvidenceConsistencyResult:
    consistent: bool
    score: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _tokens(text: str) -> Set[str]:
    stop = {"the", "and", "or", "a", "an", "of", "in", "to", "with", "for", "on", "is", "are", "has", "have", "may", "limited"}
    return {t for t in re.findall(r"[a-zA-Z0-9_+#.]+", str(text).lower()) if len(t) > 2 and t not in stop}


def _items(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, dict):
        return [str(value)]
    text = str(value).strip()
    return [text] if text else []


def _similar(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _score_value(v: Any) -> float:
    try:
        x = float(v)
        if 0 <= x <= 1:
            x *= 100
        return max(0.0, min(100.0, x))
    except Exception:
        return 0.0


def validate_evidence_consistency(output: Dict[str, Any]) -> EvidenceConsistencyResult:
    warnings: List[str] = []
    reasons = _items(output.get("reasons") or output.get("scoring_rationale") or output.get("strengths"))
    risks = _items(output.get("risks") or output.get("rejection_risks") or output.get("red_flags"))
    missing = _items(output.get("missing_evidence") or output.get("missing_skills") or output.get("weak_evidence"))
    score = _score_value(output.get("fit_score") or output.get("score") or 0)
    decision = str(output.get("shortlist_decision", "")).upper()

    if decision in {"SHORTLIST", "RECOMMEND"} and not reasons:
        warnings.append("positive_decision_without_reasons")
    if score >= 85 and len(missing) >= 3:
        warnings.append("high_score_with_many_missing_evidence_items")
    if decision == "SHORTLIST" and any("mandatory" in m.lower() or "must" in m.lower() for m in missing):
        warnings.append("shortlist_with_missing_mandatory_evidence")
    for r in risks:
        for reason in reasons:
            if _similar(r, reason) >= 0.55:
                warnings.append("risk_reason_overlap:" + r[:60])
                break
    if score <= 40 and decision == "SHORTLIST":
        warnings.append("low_score_with_shortlist_decision")
    if score >= 80 and decision == "REJECT":
        warnings.append("high_score_with_reject_decision")

    consistency_score = max(0.0, 1.0 - 0.15 * len(warnings))
    return EvidenceConsistencyResult(consistent=len(warnings) == 0, score=round(consistency_score, 4), warnings=warnings)
