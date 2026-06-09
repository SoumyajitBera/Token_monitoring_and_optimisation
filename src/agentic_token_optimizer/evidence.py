from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Iterable


def _norm(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("ml ops", "mlops")
    text = text.replace("ci cd", "ci/cd")
    text = text.replace("retrieval augmented generation", "rag")
    text = text.replace("retrieval-augmented generation", "rag")
    text = re.sub(r"[^a-z0-9+/#. -]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Canonical evidence ontology for ProdSync-style hiring/MLOps workflows.
# This is intentionally hand-built and deterministic; it does not depend on an external model/tool.
ENTITY_ALIASES: dict[str, set[str]] = {
    # core engineering skills
    "python": {"python"},
    "fastapi": {"fastapi", "fast api"},
    "sql": {"sql"},
    "api integration": {"api integration", "apis", "api"},
    "github": {"github", "git hub"},
    # genai / retrieval
    "llm": {"llm", "llms", "large language model", "large language models", "llama", "llama-3.3", "llama 3.3"},
    "rag": {"rag", "retrieval augmented generation", "retrieval-augmented generation"},
    "langchain": {"langchain", "lang chain"},
    "milvus": {"milvus", "vector database", "vector databases", "vector db", "vector store"},
    "groq": {"groq"},
    # cloud / deployment
    "docker": {"docker", "container", "containerization"},
    "kubernetes": {"kubernetes", "k8s"},
    "gcp": {"gcp", "google cloud", "google cloud platform"},
    "cloud run": {"cloud run", "gcp cloud run"},
    "cloud deployment": {"cloud deployment", "production deployment", "deployment"},
    "ci/cd": {"ci/cd", "cicd", "ci cd", "continuous integration", "continuous deployment", "pipeline"},
    # mlops / governance / validation terms - these were previously leaking
    "mlops": {"mlops", "ml ops", "machine learning operations"},
    "model registry": {"model registry", "registry lifecycle", "model lifecycle"},
    "feature store": {"feature store", "feature stores"},
    "model drift": {"model drift", "drift monitoring", "model degradation"},
    "concept drift": {"concept drift", "data drift", "distribution shift"},
    "monitoring": {"monitoring", "observability", "production monitoring", "online monitoring", "runtime monitoring"},
    "human labels": {"human labels", "human evaluation", "human-labeled", "human labelled", "human judgement", "human review"},
    "acceptance metrics": {"acceptance metrics", "task-specific metrics", "quality metrics", "task-specific acceptance metrics", "acceptance criteria"},
    "real-world validation logs": {"real-world validation logs", "real world validation logs", "validation logs", "production validation logs"},
    # ProdSync workflow terms
    "resume": {"resume", "cv"},
    "jd": {"jd", "job description"},
    "interview": {"interview", "mock interview", "interview transcript"},
    "score": {"score", "final score", "candidate score"},
    "shortlist": {"shortlist", "shortlisted", "shortlist decision"},
    "cost awareness": {"cost awareness", "token cost", "token optimization", "cost optimization", "cost"},
    "agentic ai": {"agentic", "agentic ai", "agents", "multi-agent"},
    # user's project context
    "nlq-sql": {"nlq-sql", "natural language to sql", "nlq sql"},
    "schema grounding": {"schema grounding"},
    "sql safety": {"sql safety", "sql injection prevention"},
}

# Negative evidence / risk ontology. These terms are separated from entities because they must be
# preserved even when the optimizer compresses aggressively.
CONCERN_ALIASES: dict[str, set[str]] = {
    "limited deep mlops experience": {"limited deep mlops", "limited mlops", "shallow mlops", "deeper mlops experience", "lack of deep mlops", "weak mlops"},
    "limited kubernetes ownership": {"limited kubernetes", "no production kubernetes", "kubernetes gap", "no production kubernetes ownership", "kubernetes ownership gap", "limited kubernetes ownership"},
    "limited model registry lifecycle": {"model registry", "no model registry", "lack of model registry", "model registry lifecycle", "limited model registry", "registry lifecycle gap"},
    "limited feature store experience": {"feature store", "no feature store", "lack of feature store", "feature stores", "feature store gap"},
    "weak evidence for online monitoring": {"weak evidence for online monitoring", "online monitoring", "production monitoring", "no production monitoring", "observability gap", "weak observability", "limited monitoring"},
    "needs human labels": {"human labels", "human evaluation", "human-labeled", "human labelled", "needs human labels", "requires human labels"},
    "needs task-specific acceptance metrics": {"acceptance metrics", "task-specific acceptance metrics", "task-specific metrics", "needs task-specific acceptance metrics", "acceptance criteria"},
    "model drift validation gap": {"model drift", "drift validation", "model drift monitoring", "model drift gap"},
    "concept drift validation gap": {"concept drift", "concept drift monitoring", "concept drift gap", "data drift"},
    "synthetic benchmark data": {"synthetic data", "synthetic benchmark", "requires real-world validation", "real-world validation logs", "real world validation logs"},
}

REASON_ALIASES: dict[str, set[str]] = {
    "python fastapi sql skills": {"python", "fastapi", "sql", "backend"},
    "rag vector database experience": {"rag", "milvus", "vector database", "vector db", "retrieval"},
    "cloud run docker deployment": {"cloud run", "docker", "deployment", "gcp", "cloud deployment"},
    "api integration experience": {"api integration", "api", "fastapi", "apis"},
    "cost awareness and token optimization": {"cost awareness", "token optimization", "token cost", "cost optimization"},
    "honest gap identification": {"honestly identify gaps", "not exaggerate", "do not fabricate", "honest", "gap identification"},
    "mlops governance awareness": {"mlops", "model drift", "concept drift", "monitoring", "acceptance metrics", "human labels"},
}

PROTECTED_GOVERNANCE_TERMS = [
    "acceptance metrics",
    "task-specific acceptance metrics",
    "human labels",
    "model drift",
    "concept drift",
    "monitoring",
    "observability",
    "online monitoring",
    "production monitoring",
    "real-world validation logs",
]


@dataclass
class EvidenceLedger:
    entities: list[str]
    concerns: list[str]
    reasons: list[str]

    def compact_text(self) -> str:
        return (
            "EVIDENCE_LEDGER:\n"
            f"entities: {', '.join(self.entities) if self.entities else 'N/A'}\n"
            f"concerns: {', '.join(self.concerns) if self.concerns else 'N/A'}\n"
            f"reasons: {', '.join(self.reasons) if self.reasons else 'N/A'}"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _word_boundary_contains(lower_text: str, alias: str) -> bool:
    alias = _norm(alias)
    if not alias:
        return False
    return bool(re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", lower_text))


def _contains_any(lower_text: str, aliases: Iterable[str]) -> bool:
    return any(_word_boundary_contains(lower_text, alias) for alias in aliases)


def canonical_hits(text: str, ontology: dict[str, set[str]]) -> set[str]:
    lower = _norm(text)
    return {canonical for canonical, aliases in ontology.items() if _contains_any(lower, aliases)}


def build_evidence_ledger(context: str) -> EvidenceLedger:
    lower = _norm(context)
    entities = sorted(canonical_hits(lower, ENTITY_ALIASES))
    concerns = sorted(canonical_hits(lower, CONCERN_ALIASES))
    reasons = sorted(canonical_hits(lower, REASON_ALIASES))

    # If a concern exists, include its key governance entities too. This avoids treating
    # "needs task-specific acceptance metrics" as only a concern while losing the entity.
    if "needs task-specific acceptance metrics" in concerns and "acceptance metrics" not in entities:
        entities.append("acceptance metrics")
    if "needs human labels" in concerns and "human labels" not in entities:
        entities.append("human labels")
    if "model drift validation gap" in concerns and "model drift" not in entities:
        entities.append("model drift")
    if "concept drift validation gap" in concerns and "concept drift" not in entities:
        entities.append("concept drift")

    return EvidenceLedger(
        entities=sorted(set(entities)),
        concerns=sorted(set(concerns)),
        reasons=sorted(set(reasons)),
    )


def protected_terms_from_ledger(ledger: EvidenceLedger, limit: int = 80) -> list[str]:
    # Concerns first, then governance, then entities/reasons. Negative evidence matters more
    # than generic positive skills in hiring-risk preservation.
    terms = []
    for group in [ledger.concerns, PROTECTED_GOVERNANCE_TERMS, ledger.entities, ledger.reasons]:
        for t in group:
            if t and t not in terms:
                terms.append(t)
    return terms[:limit]

# Severity weights for retention scoring. Not all concerns are equal.
# Losing "model drift" or "acceptance metrics" must hurt more than losing benchmark metadata.
CONCERN_WEIGHTS: dict[str, float] = {
    "limited deep mlops experience": 3.0,
    "limited kubernetes ownership": 3.0,
    "limited model registry lifecycle": 2.5,
    "limited feature store experience": 2.5,
    "weak evidence for online monitoring": 3.0,
    "needs human labels": 2.5,
    "needs task-specific acceptance metrics": 3.0,
    "model drift validation gap": 3.0,
    "concept drift validation gap": 3.0,
    "synthetic benchmark data": 0.5,
}
