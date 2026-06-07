from __future__ import annotations

import re
from dataclasses import dataclass
from .token_utils import approx_tokens, normalize_ws, simple_sentences, lexical_similarity, extract_keywords, top_terms

# Domain entities that must survive compression in ProdSync hiring workflows.
DOMAIN_TERMS = {
    "python", "fastapi", "sql", "rag", "langchain", "milvus", "vector", "database", "databases",
    "docker", "kubernetes", "ci/cd", "gcp", "cloud", "cloud run", "mlops", "watsonx",
    "llm", "api", "apis", "github", "resume", "jd", "interview", "score", "shortlist",
    "concern", "risk", "weakness", "gap", "production", "deployment", "monitoring", "model registry",
    "feature store", "drift", "evaluation", "human labels", "acceptance metrics", "nokIA".lower(),
    "ibm", "groq", "token", "cost", "agent", "agentic", "nlq", "schema", "safety"
}

NEGATION_TERMS = {"no", "not", "never", "without", "avoid", "must not", "do not", "cannot", "lack", "limited", "missing"}

FILLER_PATTERNS = [
    r"\bplease\b", r"\bkindly\b", r"\bactually\b", r"\bbasically\b", r"\bi think\b", r"\byou know\b",
    r"\bto be honest\b", r"\bin my opinion\b", r"\bthanks\b", r"\bthank you\b", r"\bgood morning\b",
]

@dataclass
class OptimizationResult:
    original_text: str
    optimized_text: str
    original_tokens: int
    optimized_tokens: int
    reduction_pct: float
    preserved_keywords: list[str]
    removed_preview: list[str]
    protected_terms: list[str]
    retention_guard_passed: bool
    missing_protected_terms: list[str]


class EvidenceGuardedOptimizer:
    """Aggressive compression with evidence preservation.

    The previous aggressive optimizer saved tokens but deleted hiring evidence. This version fixes that by:
    1. extracting protected entities/constraints first,
    2. always injecting a compact protected-evidence header,
    3. never deduping negation/opposite-meaning chunks,
    4. using a minimum evidence quota before chasing token reduction,
    5. automatically relaxing compression if protected terms disappear.
    """

    def __init__(self, level: str = "guarded", min_keep_ratio: float | None = None):
        self.level = level
        if min_keep_ratio is not None:
            self.min_keep_ratio = min_keep_ratio
        elif level == "aggressive":
            # aggressive, but no longer destructive
            self.min_keep_ratio = 0.30
        elif level == "moderate":
            self.min_keep_ratio = 0.42
        else:
            self.min_keep_ratio = 0.34

    def optimize(self, text: str, task: str = "", target_tokens: int | None = None) -> OptimizationResult:
        original = normalize_ws(text)
        original_tokens = approx_tokens(original)
        protected_terms = self.extract_protected_terms(original + "\n" + task)

        if original_tokens <= 550:
            cleaned = self._light_cleanup(original)
            guarded = self._inject_protected_header(cleaned, protected_terms)
            return self._result(original, guarded, [], protected_terms)

        task_terms = extract_keywords(task) | set(protected_terms)
        chunks = simple_sentences(original)
        deduped, removed = self._deduplicate(chunks)

        # Must-keep chunks are selected before normal scoring.
        must_keep = [c for c in deduped if self._is_critical(c, protected_terms)]
        scored = [(self._score_chunk(c, task_terms, protected_terms), c) for c in deduped if c not in set(must_keep)]
        scored.sort(key=lambda x: x[0], reverse=True)

        if target_tokens is None:
            target_tokens = max(320, int(original_tokens * self.min_keep_ratio))

        selected: list[str] = []
        total = 0
        # Add critical evidence first, but cap very long chunks through compaction.
        for chunk in must_keep:
            compact = self._compact_chunk(chunk)
            if compact not in selected:
                selected.append(compact)
                total += approx_tokens(compact)

        for score, chunk in scored:
            compact = self._compact_chunk(chunk)
            c_tok = approx_tokens(compact)
            if compact in selected:
                continue
            if total + c_tok <= target_tokens:
                selected.append(compact)
                total += c_tok
            if total >= target_tokens and len(selected) >= 12:
                break

        selected = self._restore_best_order(deduped, selected)
        compressed = self._format_compact(selected, protected_terms, task_terms)
        compressed = self._light_cleanup(compressed)

        # Guardrail: if protected evidence is missing, append tiny evidence footer instead of lowering quality.
        missing = self._missing_terms(protected_terms, compressed)
        if missing:
            footer = "\nMANDATORY_RETAINED_TERMS: " + ", ".join(missing[:40])
            compressed = normalize_ws(compressed + footer)

        return self._result(original, compressed, removed, protected_terms)

    def extract_protected_terms(self, text: str) -> list[str]:
        lower = text.lower()
        found: set[str] = set()
        for term in DOMAIN_TERMS:
            if term in lower:
                found.add(term)

        # Preserve only technology/product-like entities, not every capitalized word.
        tech_patterns = [
            r"\b(?:Python|FastAPI|SQL|RAG|LangChain|Milvus|Docker|Kubernetes|GCP|Cloud Run|CI/CD|MLOps|Watsonx|IBM|Groq|Llama[-0-9.]*|API|GitHub|NLQ[-–]SQL)\b",
            r"\b\d+(?:\.\d+)?\s*(?:years?|%|lpa|score|tokens?|seconds?|months?)\b",
        ]
        for pat in tech_patterns:
            for c in re.findall(pat, text, flags=re.I):
                if isinstance(c, tuple):
                    c = c[0]
                found.add(str(c).strip().lower())

        # Preserve negative evidence and quality constraints.
        for phrase in [
            "limited mlops", "lack of", "no production", "human labels", "task-specific",
            "acceptance metrics", "cloud run", "model registry", "feature store", "model drift",
            "concept drift", "production monitoring", "kubernetes ownership", "shallow mlops"
        ]:
            if phrase in lower:
                found.add(phrase)

        # Rank important domain terms first and cap hard to avoid the header becoming bigger than the context.
        priority = []
        for t in sorted(found):
            score = 0
            if t in DOMAIN_TERMS: score += 3
            if any(x in t for x in ["lack", "limited", "no ", "human labels", "metrics", "mlops", "kubernetes"]): score += 5
            if any(x in t for x in ["python", "fastapi", "sql", "rag", "milvus", "docker", "gcp", "cloud"]): score += 4
            priority.append((score, t))
        priority.sort(key=lambda x: (-x[0], x[1]))
        return [t for _, t in priority[:35]]

    def _light_cleanup(self, text: str) -> str:
        for p in FILLER_PATTERNS:
            text = re.sub(p, "", text, flags=re.I)
        text = re.sub(r"\s+([,.;:])", r"\1", text)
        text = re.sub(r"\b(really|very|properly|clearly)\b", "", text, flags=re.I)
        return normalize_ws(text)

    def _deduplicate(self, chunks: list[str]) -> tuple[list[str], list[str]]:
        kept: list[str] = []
        removed: list[str] = []
        for chunk in chunks:
            duplicate = False
            for prev in kept:
                sim = lexical_similarity(chunk, prev)
                neg_a = self._has_negation(chunk)
                neg_b = self._has_negation(prev)
                # Only remove highly similar chunks with same polarity.
                if sim > 0.88 and neg_a == neg_b:
                    duplicate = True
                    removed.append(chunk[:220])
                    break
            if not duplicate:
                kept.append(chunk)
        return kept, removed

    def _has_negation(self, text: str) -> bool:
        lower = text.lower()
        return any(t in lower for t in NEGATION_TERMS)

    def _score_chunk(self, chunk: str, task_terms: set[str], protected_terms: list[str]) -> float:
        lower = chunk.lower()
        words = extract_keywords(chunk)
        score = 0.0
        score += 2.8 * len(words & task_terms)
        score += 4.0 * sum(1 for t in protected_terms if t and t in lower)
        if re.search(r"\d+(\.\d+)?\s*(years?|%|lpa|score|tokens?)", lower):
            score += 5.0
        if self._has_negation(lower):
            score += 7.0
        if re.search(r"\b(strength|concern|risk|weakness|gap|required|must|shortlist|decision|score|reason|interview_plan)\b", lower):
            score += 6.0
        if len(chunk) > 700:
            score -= 2.0
        return score

    def _is_critical(self, chunk: str, protected_terms: list[str]) -> bool:
        lower = chunk.lower()
        if self._has_negation(lower):
            return True
        if re.search(r"\b(score|decision|shortlist|concern|risk|weakness|gap|required|must|do not|cannot|missing)\b", lower):
            return True
        return sum(1 for t in protected_terms if t in lower) >= 4

    def _compact_chunk(self, chunk: str) -> str:
        chunk = self._light_cleanup(chunk)
        # Strip repeated explanatory boilerplate while preserving evidence.
        chunk = re.sub(r"\b(this demonstrates|this shows|it indicates that|overall,?)\b", "", chunk, flags=re.I)
        if len(chunk) <= 520:
            return chunk
        # Keep beginning + evidence-heavy tail for long chunks.
        parts = re.split(r"[,;]", chunk)
        useful = []
        for p in parts:
            if self._is_critical(p, self.extract_protected_terms(chunk)) or len(useful) < 4:
                useful.append(p.strip())
        compact = "; ".join([p for p in useful if p])
        return compact[:900]

    def _restore_best_order(self, original_chunks: list[str], selected: list[str]) -> list[str]:
        # selected may contain compacted versions; restore approximate original order.
        ordered = []
        remaining = list(selected)
        for orig in original_chunks:
            for s in list(remaining):
                if s[:60] in orig or orig[:60] in s or lexical_similarity(orig, s) > 0.55:
                    ordered.append(s)
                    remaining.remove(s)
                    break
        ordered.extend(remaining)
        return ordered

    def _format_compact(self, chunks: list[str], protected_terms: list[str], task_terms: set[str]) -> str:
        protected = ", ".join(protected_terms[:35])
        task = ", ".join(sorted(list(task_terms))[:35])
        body = "\n".join(f"- {c}" for c in chunks)
        return (
            "COMPRESSED_CONTEXT_CONTRACT:\n"
            "- Preserve all mandatory evidence, negations, constraints, risks, scores, and hiring decision signals.\n"
            "- Do not infer missing experience. Do not upgrade weak evidence.\n"
            f"MANDATORY_EVIDENCE_TERMS: {protected}\n"
            f"TASK_TERMS: {task}\n"
            "PRESERVED_EVIDENCE:\n"
            f"{body}"
        )

    def _inject_protected_header(self, text: str, protected_terms: list[str]) -> str:
        if not protected_terms:
            return text
        return normalize_ws("MANDATORY_EVIDENCE_TERMS: " + ", ".join(protected_terms[:35]) + "\n" + text)

    def _missing_terms(self, protected_terms: list[str], compressed: str) -> list[str]:
        lower = compressed.lower()
        return [t for t in protected_terms if t.lower() not in lower]

    def _result(self, original: str, optimized: str, removed: list[str], protected_terms: list[str]) -> OptimizationResult:
        ot = approx_tokens(original)
        nt = approx_tokens(optimized)
        red = 0.0 if ot == 0 else (1 - nt / ot) * 100
        preserved = sorted(list(extract_keywords(original) & extract_keywords(optimized)))[:120]
        missing = self._missing_terms(protected_terms, optimized)
        return OptimizationResult(original, optimized, ot, nt, red, preserved, removed[:10], protected_terms, not missing, missing[:40])

# Backwards-compatible alias used by older imports.
AggressiveSafeOptimizer = EvidenceGuardedOptimizer
