from __future__ import annotations

import json
import re
from dataclasses import dataclass
from .token_utils import lexical_similarity
from .semantic_similarity import deterministic_semantic_similarity
from .evidence import (
    ENTITY_ALIASES,
    CONCERN_ALIASES,
    REASON_ALIASES,
    CONCERN_WEIGHTS,
    canonical_hits,
    build_evidence_ledger,
)


@dataclass
class QualityReport:
    decision_preserved: bool
    score_delta: float | None
    keyword_retention_pct: float
    entity_retention_pct: float
    concern_retention_pct: float
    raw_concern_retention_pct: float
    weighted_concern_retention_pct: float
    reason_overlap_pct: float
    output_similarity: float
    semantic_similarity: float
    semantic_similarity_pct: float
    semantic_similarity_details: dict[str, float]
    semantic_pass: bool
    semantic_formula: str
    overall_retention_pct: float
    baseline_decision: str | None
    optimized_decision: str | None
    baseline_score: float | None
    optimized_score: float | None
    missing_entities: list[str]
    missing_concerns: list[str]
    missing_concern_weights: dict[str, float]
    score_preserved: bool
    retention_pass: bool
    retention_formula: str
    concern_weight_formula: str


def _json(text: str) -> dict:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}
        return {}


def _norm(value: str) -> str:
    value = value.lower().strip()
    value = value.replace("ml ops", "mlops")
    value = value.replace("ci cd", "ci/cd")
    value = value.replace("retrieval augmented generation", "rag")
    value = value.replace("retrieval-augmented generation", "rag")
    value = re.sub(r"[^a-z0-9+/#. -]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _normalize_decision(value: str | None) -> str | None:
    if not value:
        return None
    v = _norm(value)
    if v in {"yes", "shortlist", "shortlisted", "selected", "recommend", "recommended", "hire", "proceed"}:
        return "shortlist"
    if v in {"no", "reject", "rejected", "do not shortlist", "not shortlist"}:
        return "reject"
    if v in {"maybe", "hold", "borderline", "manual review", "review", "n/a", "na"}:
        return "hold" if v != "n/a" and v != "na" else None
    if "shortlist" in v or "recommend" in v or "proceed" in v:
        return "shortlist"
    if "reject" in v or "do not" in v:
        return "reject"
    if "hold" in v or "review" in v:
        return "hold"
    return v or None


def _decision(text: str) -> str | None:
    obj = _json(text)
    for key in ["shortlist_decision", "decision", "recommendation"]:
        if obj.get(key) is not None:
            return _normalize_decision(str(obj.get(key)))
    patterns = [
        r"shortlist_decision\s*[:=]\s*['\"]?([A-Za-z _/-]+)",
        r"decision\s*[:=]\s*['\"]?([A-Za-z _/-]+)",
        r"Final\s+Shortlist\s+Decision.*?(Shortlist|Reject|Hold|Yes|No|Maybe)",
    ]
    for p in patterns:
        m = re.search(p, text, re.I | re.S)
        if m:
            raw = m.group(1).strip().split("\n")[0].strip(" -*.,'")
            return _normalize_decision(raw)
    return None


def _score(text: str) -> float | None:
    obj = _json(text)
    for key in ["final_score", "score", "candidate_score"]:
        try:
            if obj.get(key) is not None:
                return float(obj.get(key))
        except Exception:
            pass
    patterns = [
        r"final_score\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        r"final[_ ]?score\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        r"score\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            return float(m.group(1))
    return None


def _collect_evidence_text(text: str) -> str:
    obj = _json(text)
    if not obj:
        return text
    fields = [
        "evidence_terms",
        "reasons",
        "concerns",
        "risk_flags",
        "interview_plan",
        "raw_output",
    ]
    pieces = [str(obj.get(k, "")) for k in fields]
    pieces.append(text)
    return "\n".join(pieces)


def _tokens(phrase: str) -> set[str]:
    stop = {"the", "a", "an", "for", "of", "and", "or", "with", "to", "in", "on", "by", "needs", "limited", "weak"}
    return {t for t in re.findall(r"[a-z0-9+#/.-]+", _norm(phrase)) if t not in stop and len(t) > 1}


def _soft_contains(canonical: str, optimized_text: str, ontology: dict[str, set[str]]) -> bool:
    """Deterministic semantic-lite containment.

    Built from scratch: alias matching + token-overlap fallback.
    No embedding model, no external tool, no LLM judge.
    """
    lower = _norm(optimized_text)
    aliases = ontology.get(canonical, {canonical}) | {canonical}
    for alias in aliases:
        a = _norm(alias)
        if re.search(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])", lower):
            return True
    can_tokens = _tokens(canonical)
    if not can_tokens:
        return False
    opt_tokens = _tokens(lower)
    overlap = len(can_tokens & opt_tokens) / max(1, len(can_tokens))
    return overlap >= 0.67


def _category_retention(base: set[str], optimized_text: str, ontology: dict[str, set[str]]) -> tuple[float, list[str], set[str]]:
    if not base:
        return 100.0, [], set()
    retained = {item for item in base if _soft_contains(item, optimized_text, ontology)}
    missing = sorted(base - retained)
    return 100.0 * len(retained) / len(base), missing, retained


def _weighted_concern_retention(base: set[str], retained: set[str]) -> tuple[float, dict[str, float]]:
    """Severity-weighted concern retention.

    Formula:
        WCR = 100 * sum(w_i for retained concerns) / sum(w_i for baseline concerns)

    Critical risks such as model/concept drift, monitoring, acceptance metrics, and MLOps gaps
    carry more weight than benchmark metadata like "synthetic benchmark data".
    """
    if not base:
        return 100.0, {}
    weights = {c: float(CONCERN_WEIGHTS.get(c, 1.0)) for c in base}
    total_weight = sum(weights.values()) or 1.0
    retained_weight = sum(weights[c] for c in retained if c in weights)
    missing_weights = {c: weights[c] for c in sorted(base - retained)}
    return 100.0 * retained_weight / total_weight, missing_weights


def _evidence_sets(text: str) -> tuple[set[str], set[str], set[str]]:
    evidence_text = _collect_evidence_text(text)
    ledger = build_evidence_ledger(evidence_text)
    entities = set(ledger.entities) | canonical_hits(evidence_text, ENTITY_ALIASES)
    concerns = set(ledger.concerns) | canonical_hits(evidence_text, CONCERN_ALIASES)
    reasons = set(ledger.reasons) | canonical_hits(evidence_text, REASON_ALIASES)
    return entities, concerns, reasons


def _bounded_score_preservation(delta: float | None) -> tuple[float, bool]:
    if delta is None:
        return 70.0, False
    if delta <= 0.25:
        return 100.0, True
    if delta <= 0.5:
        return 85.0, False
    return max(0.0, 100.0 - delta * 25.0), False


def compare_outputs(baseline: str, optimized: str) -> QualityReport:
    bd, od = _decision(baseline), _decision(optimized)
    decision_preserved = bool(bd and od and bd == od)
    bs, os = _score(baseline), _score(optimized)
    score_delta = abs(bs - os) if bs is not None and os is not None else None
    score_component, score_preserved = _bounded_score_preservation(score_delta)

    base_entities, base_concerns, base_reasons = _evidence_sets(baseline)
    opt_text = _collect_evidence_text(optimized)

    entity_retention, missing_entities, _ = _category_retention(base_entities, opt_text, ENTITY_ALIASES)
    raw_concern_retention, missing_concerns, retained_concerns = _category_retention(base_concerns, opt_text, CONCERN_ALIASES)
    weighted_concern_retention, missing_concern_weights = _weighted_concern_retention(base_concerns, retained_concerns)
    reason_overlap, _, _ = _category_retention(base_reasons, opt_text, REASON_ALIASES)

    evidence_baseline_text = _collect_evidence_text(baseline)
    sim = lexical_similarity(evidence_baseline_text, opt_text)
    semantic_sim, semantic_details = deterministic_semantic_similarity(evidence_baseline_text, opt_text)
    semantic_pass = semantic_sim >= 0.90

    decision_component = 100.0 if decision_preserved else 0.0
    # Use weighted concern retention in the final score. This fixes the previous flaw where
    # a missing minor metadata concern hurt as much as losing a critical production-risk term.
    overall = (
        0.30 * decision_component
        + 0.15 * score_component
        + 0.20 * entity_retention
        + 0.25 * weighted_concern_retention
        + 0.10 * reason_overlap
    )
    retention_pass = (
        decision_preserved
        and score_preserved
        and entity_retention >= 90.0
        and weighted_concern_retention >= 90.0
        and overall >= 90.0
    )

    return QualityReport(
        decision_preserved=decision_preserved,
        score_delta=score_delta,
        keyword_retention_pct=entity_retention,
        entity_retention_pct=entity_retention,
        concern_retention_pct=weighted_concern_retention,
        raw_concern_retention_pct=raw_concern_retention,
        weighted_concern_retention_pct=weighted_concern_retention,
        reason_overlap_pct=reason_overlap,
        output_similarity=sim,
        semantic_similarity=semantic_sim,
        semantic_similarity_pct=semantic_sim * 100.0,
        semantic_similarity_details=semantic_details,
        semantic_pass=semantic_pass,
        semantic_formula="0.45*token_cosine + 0.25*char4_cosine + 0.30*ontology_concept_jaccard",
        overall_retention_pct=overall,
        baseline_decision=bd,
        optimized_decision=od,
        baseline_score=bs,
        optimized_score=os,
        missing_entities=missing_entities[:40],
        missing_concerns=missing_concerns[:40],
        missing_concern_weights=missing_concern_weights,
        score_preserved=score_preserved,
        retention_pass=retention_pass,
        retention_formula="0.30*decision + 0.15*score + 0.20*entity + 0.25*weighted_concern + 0.10*reason",
        concern_weight_formula="100 * retained_concern_weight / baseline_concern_weight; critical=3.0, important=2.5, minor metadata=0.5, default=1.0",
    )
