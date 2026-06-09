from __future__ import annotations

import math
import re
from collections import Counter
from .evidence import ENTITY_ALIASES, CONCERN_ALIASES, REASON_ALIASES, canonical_hits

_STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "of", "to", "in", "on", "with", "by", "from",
    "this", "that", "these", "those", "is", "are", "was", "were", "be", "been", "being",
    "candidate", "agent", "final", "score", "decision", "shortlist", "reasons", "concerns",
}

_SYNONYMS = {
    "genai": "llm",
    "large language model": "llm",
    "large language models": "llm",
    "retrieval augmented generation": "rag",
    "retrieval-augmented generation": "rag",
    "vector database": "vector db",
    "vector databases": "vector db",
    "observability": "monitoring",
    "online monitoring": "monitoring",
    "production monitoring": "monitoring",
    "evaluation metric": "acceptance metrics",
    "evaluation metrics": "acceptance metrics",
    "task specific acceptance metrics": "acceptance metrics",
    "task-specific acceptance metrics": "acceptance metrics",
    "human label": "human labels",
    "manual labels": "human labels",
    "k8s": "kubernetes",
    "ml ops": "mlops",
    "ci cd": "ci/cd",
}


def _norm(text: str) -> str:
    text = text.lower()
    text = text.replace("ml ops", "mlops").replace("ci cd", "ci/cd")
    for old, new in sorted(_SYNONYMS.items(), key=lambda x: len(x[0]), reverse=True):
        text = re.sub(r"(?<![a-z0-9])" + re.escape(old) + r"(?![a-z0-9])", new, text)
    text = re.sub(r"[^a-z0-9+/#. -]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9+/#.-]+", _norm(text))
    return [t for t in toks if t not in _STOPWORDS and len(t) > 1]


def _char_ngrams(text: str, n: int = 4) -> list[str]:
    compact = re.sub(r"\s+", " ", _norm(text))
    if len(compact) < n:
        return [compact] if compact else []
    return [compact[i : i + n] for i in range(len(compact) - n + 1)]


def _cosine_counter(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _concepts(text: str) -> set[str]:
    ont = {}
    ont.update(ENTITY_ALIASES)
    ont.update(CONCERN_ALIASES)
    ont.update(REASON_ALIASES)
    hits = set()
    hits |= canonical_hits(text, ENTITY_ALIASES)
    hits |= canonical_hits(text, CONCERN_ALIASES)
    hits |= canonical_hits(text, REASON_ALIASES)
    # Add high-signal normalized phrases that may not be in ontology.
    normalized = _norm(text)
    for phrase in _SYNONYMS.values():
        if phrase in normalized:
            hits.add(phrase)
    return hits


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def deterministic_semantic_similarity(a: str, b: str) -> tuple[float, dict[str, float]]:
    """Deterministic semantic similarity built from scratch.

    No LLM judge, no embedding API, no external semantic service.

    Formula:
        S = 0.45 * cosine(TF_token(a), TF_token(b))
          + 0.25 * cosine(TF_char4(a), TF_char4(b))
          + 0.30 * Jaccard(ontology_concepts(a), ontology_concepts(b))

    Why this is better than raw lexical/Jaccard:
    - aliases are normalized before scoring;
    - short wording changes are absorbed by char-gram cosine;
    - evidence ontology terms carry explicit semantic weight.
    """
    token_cos = _cosine_counter(Counter(_tokens(a)), Counter(_tokens(b)))
    char_cos = _cosine_counter(Counter(_char_ngrams(a)), Counter(_char_ngrams(b)))
    concept_j = _jaccard(_concepts(a), _concepts(b))
    score = (0.45 * token_cos) + (0.25 * char_cos) + (0.30 * concept_j)
    details = {
        "token_cosine": token_cos,
        "char4_cosine": char_cos,
        "concept_jaccard": concept_j,
    }
    return max(0.0, min(1.0, score)), details
