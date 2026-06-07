from __future__ import annotations

import re
from dataclasses import dataclass
from .token_utils import extract_keywords, lexical_similarity

# Important hiring/domain terms only. Generic English overlap is trash for this use case.
ENTITY_ALIASES: dict[str, set[str]] = {
    "python": {"python"},
    "fastapi": {"fastapi"},
    "sql": {"sql"},
    "rag": {"rag", "retrieval augmented generation", "retrieval-augmented generation"},
    "langchain": {"langchain"},
    "milvus": {"milvus", "vector database", "vector databases", "vector db"},
    "docker": {"docker"},
    "kubernetes": {"kubernetes", "k8s"},
    "gcp": {"gcp", "google cloud"},
    "cloud run": {"cloud run"},
    "cloud deployment": {"cloud deployment", "deployment", "production deployment", "cloud"},
    "ci/cd": {"ci/cd", "cicd", "continuous integration", "continuous deployment"},
    "mlops": {"mlops", "ml ops"},
    "model registry": {"model registry"},
    "feature store": {"feature store", "feature stores"},
    "model drift": {"model drift", "concept drift", "drift"},
    "monitoring": {"monitoring", "observability", "production monitoring"},
    "human labels": {"human labels", "human evaluation", "human-labeled", "human labelled"},
    "acceptance metrics": {"acceptance metrics", "task-specific metrics", "quality metrics"},
    "groq": {"groq"},
    "llm": {"llm", "large language model"},
    "api": {"api", "apis", "api integration"},
    "github": {"github"},
    "resume": {"resume"},
    "jd": {"jd", "job description"},
    "interview": {"interview"},
    "shortlist": {"shortlist", "selected", "recommended", "proceed"},
    "gap": {"gap", "gaps", "weakness", "weaknesses", "limited", "lack"},
    "cost awareness": {"cost", "cost awareness", "token cost", "token optimization"},
    "agentic ai": {"agentic", "agentic ai", "agents"},
    "nlq-sql": {"nlq-sql", "natural language to sql", "nlq sql"},
    "schema grounding": {"schema grounding", "schema"},
    "sql safety": {"sql safety", "safety"},
}

NEGATIVE_EVIDENCE = {
    "limited mlops", "lack of deep mlops", "no production kubernetes", "limited kubernetes",
    "no model registry", "lack of model registry", "no feature store", "lack of feature store",
    "no production monitoring", "shallow mlops", "synthetic data", "needs real-world validation",
    "human labels", "acceptance metrics"
}

@dataclass
class QualityReport:
    decision_preserved: bool
    score_delta: float | None
    keyword_retention_pct: float
    entity_retention_pct: float
    concern_retention_pct: float
    reason_overlap_pct: float
    output_similarity: float
    overall_retention_pct: float
    baseline_decision: str | None
    optimized_decision: str | None
    baseline_score: float | None
    optimized_score: float | None
    missing_entities: list[str]
    missing_concerns: list[str]


def _normalize_decision(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip().lower()
    v = re.sub(r"[^a-z ]", "", v).strip()
    if v in {"yes", "shortlist", "selected", "recommend", "recommended", "hire", "proceed"}:
        return "shortlist"
    if v in {"no", "reject", "rejected", "do not shortlist", "not shortlist"}:
        return "reject"
    if v in {"maybe", "hold", "borderline", "manual review", "review"}:
        return "hold"
    if "shortlist" in v or "recommend" in v:
        return "shortlist"
    if "reject" in v or "do not" in v:
        return "reject"
    return v or None


def _decision(text: str) -> str | None:
    patterns = [
        r"shortlist_decision\s*[:=]\s*['\"]?([A-Za-z _-]+)",
        r"\*\*\s*decision\s*:\s*\*\*\s*([A-Za-z _-]+)",
        r"decision\s*[:=]\s*['\"]?([A-Za-z _-]+)",
        r"Final\s+Shortlist\s+Decision.*?(Shortlist|Reject|Hold|Yes|No|Maybe)",
    ]
    for p in patterns:
        m = re.search(p, text, re.I | re.S)
        if m:
            raw = m.group(1).strip().split("\n")[0].strip(" -*.,'")
            return _normalize_decision(raw)
    return None


def _score(text: str) -> float | None:
    patterns = [
        r"final_score\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        r"\*\*\s*final[_ ]?score\s*:\s*\*\*\s*([0-9]+(?:\.[0-9]+)?)",
        r"final[_ ]?score\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            return float(m.group(1))
    return None


def _entities(text: str) -> set[str]:
    lower = text.lower()
    found = set()
    # Exact alias matching with normalization.
    for canonical, aliases in ENTITY_ALIASES.items():
        for alias in aliases:
            if re.search(r"(?<![a-z0-9])" + re.escape(alias.lower()) + r"(?![a-z0-9])", lower):
                found.add(canonical)
                break
    # Preserve numeric evidence too.
    for m in re.findall(r"\b\d+(?:\.\d+)?\s*(?:years?|%|lpa|score|tokens?|seconds?|months?)\b", lower):
        found.add(m.strip())
    return found


def _concerns(text: str) -> set[str]:
    lower = text.lower()
    found = set()
    for c in NEGATIVE_EVIDENCE:
        if c in lower:
            found.add(c)
    # Also capture concise concern lines.
    block = re.search(r"concerns\s*:\s*(.*?)(?:interview_plan|evidence_terms|$)", text, flags=re.I | re.S)
    if block:
        for line in re.split(r"[\n;,\[\]]+", block.group(1)):
            clean = re.sub(r"[^a-z0-9 /-]", " ", line.lower()).strip()
            if any(x in clean for x in ["limited", "lack", "no ", "missing", "weak", "gap"]):
                if 3 <= len(clean.split()) <= 10:
                    found.add(clean)
    return found


def _bullet_terms(text: str, label: str) -> set[str]:
    m = re.search(label + r"\s*:\s*(.*?)(?:\n[a-z_ ]+\s*:|$)", text, flags=re.I | re.S)
    if not m:
        return set()
    return extract_keywords(m.group(1)) - {"candidate", "experience", "skills", "strong", "relevant"}


def _retention(base: set[str], opt: set[str]) -> tuple[float, list[str]]:
    if not base:
        return 100.0, []
    missing = sorted(list(base - opt))
    return 100.0 * len(base & opt) / len(base), missing


def compare_outputs(baseline: str, optimized: str) -> QualityReport:
    bd, od = _decision(baseline), _decision(optimized)
    decision_preserved = bool(bd and od and bd == od)
    bs, os = _score(baseline), _score(optimized)
    score_delta = abs(bs - os) if bs is not None and os is not None else None

    base_entities = _entities(baseline)
    opt_entities = _entities(optimized)
    entity_retention, missing_entities = _retention(base_entities, opt_entities)

    base_concerns = _concerns(baseline)
    opt_concerns = _concerns(optimized)
    concern_retention, missing_concerns = _retention(base_concerns, opt_concerns)

    base_reason_terms = _bullet_terms(baseline, "reasons")
    opt_reason_terms = _bullet_terms(optimized, "reasons")
    reason_overlap, _ = _retention(base_reason_terms, opt_reason_terms)

    # Keep old name for continuity, but now it means important evidence-keyword retention, not dumb word overlap.
    keyword_retention = entity_retention
    sim = lexical_similarity(baseline, optimized)
    score_component = 100.0 if score_delta is None else max(0.0, 100.0 - score_delta * 12.0)
    decision_component = 100.0 if decision_preserved else 0.0
    overall = (
        0.35 * decision_component
        + 0.20 * score_component
        + 0.25 * entity_retention
        + 0.15 * concern_retention
        + 0.05 * reason_overlap
    )
    return QualityReport(
        decision_preserved=decision_preserved,
        score_delta=score_delta,
        keyword_retention_pct=keyword_retention,
        entity_retention_pct=entity_retention,
        concern_retention_pct=concern_retention,
        reason_overlap_pct=reason_overlap,
        output_similarity=sim,
        overall_retention_pct=overall,
        baseline_decision=bd,
        optimized_decision=od,
        baseline_score=bs,
        optimized_score=os,
        missing_entities=missing_entities[:25],
        missing_concerns=missing_concerns[:25],
    )
