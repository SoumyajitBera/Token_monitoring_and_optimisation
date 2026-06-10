from dataclasses import asdict, dataclass, field
from typing import Dict, List

from ..extractors import extract_codes, extract_entities, extract_keywords, extract_numbers
from ..math_utils import words
from ..tokenization import count_tokens, split_sentences


@dataclass
class ProdSyncCompactMemory:
    candidate_summary: str
    jd_summary: str
    core_skills: List[str] = field(default_factory=list)
    role_requirements: List[str] = field(default_factory=list)
    numeric_facts: List[str] = field(default_factory=list)
    named_entities: List[str] = field(default_factory=list)
    safety_constraints: List[str] = field(default_factory=list)
    evidence_digest: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def to_context_block(self) -> str:
        return (
            "[PRODSYNC_COMPACT_MEMORY]\n"
            f"candidate_summary: {self.candidate_summary}\n"
            f"jd_summary: {self.jd_summary}\n"
            f"core_skills: {', '.join(self.core_skills[:35])}\n"
            f"role_requirements: {', '.join(self.role_requirements[:35])}\n"
            f"numeric_facts: {', '.join(self.numeric_facts[:25])}\n"
            f"named_entities: {', '.join(self.named_entities[:25])}\n"
            f"safety_constraints: {'; '.join(self.safety_constraints[:20])}\n"
            f"evidence_digest: {'; '.join(self.evidence_digest[:20])}\n"
        )


def _top_sentences(text: str, query_terms: List[str], limit: int = 6) -> List[str]:
    sentences = split_sentences(text or "")
    scored = []
    q = set(t.lower() for t in query_terms)
    for s in sentences:
        toks = set(words(s))
        score = len(toks & q) + min(8, len(extract_numbers(s))) * 2 + min(4, len(extract_codes(s))) * 2
        if any(x in s.lower() for x in ["must", "required", "mandatory", "experience", "skill", "project", "rag", "llm", "python", "fastapi", "kubernetes", "vector"]):
            score += 3
        scored.append((score, s.strip()))
    selected = [s for _, s in sorted(scored, key=lambda x: (-x[0], len(x[1])))[:limit] if s]
    return selected


def _skills_from_text(text: str, limit: int = 35) -> List[str]:
    known = [
        "python", "fastapi", "django", "flask", "sql", "postgres", "tidb", "mysql", "mongodb",
        "machine learning", "deep learning", "nlp", "computer vision", "llm", "rag", "agents", "agentic",
        "langchain", "crewai", "milvus", "vector search", "sentence transformers", "embeddings", "groq",
        "watsonx", "openai", "api", "cloud run", "gcp", "docker", "kubernetes", "ci/cd", "mlops",
        "monitoring", "cost optimization", "prompt optimization", "retention scoring", "evaluation", "governance"
    ]
    low = (text or "").lower()
    hits = []
    for k in known:
        if k in low:
            hits.append(k)
    for kw in extract_keywords(text, max_keywords=60):
        if kw not in hits and len(kw) > 3:
            hits.append(kw)
        if len(hits) >= limit:
            break
    return hits[:limit]


def _constraints_from_text(text: str, limit: int = 20) -> List[str]:
    out = []
    for s in split_sentences(text or ""):
        low = s.lower()
        if any(x in low for x in ["must", "required", "mandatory", "should", "only", "unless", "without", "not", "minimum", "maximum", "at least", "json", "schema"]):
            out.append(s.strip())
        if len(out) >= limit:
            break
    return out


def build_prodsync_compact_memory(candidate_profile: str, job_description: str, evidence_context: List[str]) -> ProdSyncCompactMemory:
    all_text = "\n".join([candidate_profile or "", job_description or ""] + (evidence_context or []))
    candidate_terms = _skills_from_text(candidate_profile, limit=25)
    jd_terms = _skills_from_text(job_description, limit=30)
    candidate_summary = " | ".join(_top_sentences(candidate_profile, candidate_terms + ["project", "experience", "skill"], limit=5))
    jd_summary = " | ".join(_top_sentences(job_description, jd_terms + ["requirement", "must", "experience"], limit=5))

    evidence = []
    for item in evidence_context or []:
        evidence.extend(_top_sentences(item, candidate_terms + jd_terms, limit=2))
    # Keep memory compact and bounded.
    evidence = evidence[:12]

    memory = ProdSyncCompactMemory(
        candidate_summary=candidate_summary[:1200],
        jd_summary=jd_summary[:1200],
        core_skills=candidate_terms,
        role_requirements=jd_terms,
        numeric_facts=sorted(set(extract_numbers(all_text)))[:25],
        named_entities=sorted(set(extract_entities(all_text) + extract_codes(all_text)))[:25],
        safety_constraints=_constraints_from_text(all_text, limit=18),
        evidence_digest=evidence,
    )
    return memory


def route_context_for_agent(
    agent_name: str,
    memory: ProdSyncCompactMemory,
    raw_candidate: str,
    raw_jd: str,
    evidence_context: List[str],
    previous_outputs: List[str],
) -> List[str]:
    """Route only necessary context to each agent.

    This is the key cost-refinement layer: avoid sending full resume + full JD
    + every previous output to every agent.
    """
    mem = memory.to_context_block()
    prev_compact = previous_outputs[-2:] if previous_outputs else []

    if agent_name == "ResumeIntelligenceAgent":
        return [mem, "[RAW_CANDIDATE_PROFILE]\n" + raw_candidate[:4500]] + (evidence_context[:2] if evidence_context else [])
    if agent_name == "JDRequirementAgent":
        return [mem, "[RAW_JOB_DESCRIPTION]\n" + raw_jd[:4500]]
    if agent_name == "SemanticFitScoringAgent":
        return [mem] + prev_compact[-2:]
    if agent_name == "SkillGapReadinessAgent":
        return [mem] + prev_compact[-2:]
    if agent_name == "InterviewQuestionAgent":
        # Most compressible agent: no full raw resume/JD, only compact memory + last outputs.
        return [mem] + prev_compact[-2:]
    if agent_name == "RecruiterDecisionAgent":
        return [mem] + previous_outputs[-4:]
    return [mem] + prev_compact


def memory_stats(memory: ProdSyncCompactMemory, raw_candidate: str, raw_jd: str, evidence_context: List[str]) -> Dict[str, object]:
    raw = "\n".join([raw_candidate or "", raw_jd or ""] + (evidence_context or []))
    compact = memory.to_context_block()
    raw_tokens = count_tokens(raw)
    compact_tokens = count_tokens(compact)
    return {
        "raw_memory_tokens": raw_tokens,
        "compact_memory_tokens": compact_tokens,
        "memory_tokens_saved": max(0, raw_tokens - compact_tokens),
        "memory_reduction_percentage": round(((raw_tokens - compact_tokens) / raw_tokens) * 100, 4) if raw_tokens else 0.0,
    }
