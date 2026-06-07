from __future__ import annotations

from dataclasses import dataclass
from .llm_client import GroqClient, LLMResponse
from .optimizer import AggressiveSafeOptimizer, OptimizationResult, EvidenceGuardedOptimizer
from .costing import estimate_cost_usd

@dataclass
class AgentResult:
    name: str
    baseline_output: str | None
    optimized_output: str | None
    baseline_input_tokens: int
    baseline_output_tokens: int
    optimized_input_tokens: int
    optimized_output_tokens: int
    original_context_tokens: int
    optimized_context_tokens: int
    context_reduction_pct: float
    cost_baseline: float
    cost_optimized: float

class ProdSyncAgent:
    def __init__(self, name: str, task: str):
        self.name = name
        self.task = task
        self._term_extractor = EvidenceGuardedOptimizer(level="guarded")

    def system_prompt(self) -> str:
        return (
            f"You are {self.name} inside ProdSync, an agentic hiring intelligence system. "
            "Return compact structured output only. No greeting. No conclusion. "
            "Never invent experience. Preserve factual constraints, risks, scores, tools, gaps, and hiring decision evidence. "
            "Use exact technology names from MANDATORY_EVIDENCE_TERMS whenever they are supported by context. "
            "Keep concerns/risks even when compressing. Use the exact schema requested by the user."
        )

    def _mandatory_terms(self, context: str) -> str:
        terms = self._term_extractor.extract_protected_terms(context + "\n" + self.task)
        # Give the LLM a controlled vocabulary so baseline and optimized outputs keep the same important terms.
        return ", ".join(terms[:45]) if terms else "N/A"

    def user_prompt(self, context: str) -> str:
        terms = self._mandatory_terms(context)
        return (
            f"ACTIVE_AGENT: {self.name}\n"
            f"TASK: {self.task}\n"
            f"MANDATORY_EVIDENCE_TERMS: {terms}\n"
            "RULES:\n"
            "- Use MANDATORY_EVIDENCE_TERMS exactly when relevant.\n"
            "- Do not drop negative evidence such as limited MLOps, missing Kubernetes, no model registry, no feature store, no monitoring, or need for human labels.\n"
            "- Keep same meaning even if context is compressed.\n"
            "- If evidence is absent, say N/A; do not fabricate.\n"
            "CONTEXT:\n"
            f"{context}\n\n"
            "Return only this schema:\n"
            "final_score: <0-10 or N/A>\n"
            "shortlist_decision: <Shortlist/Reject/Hold/N/A>\n"
            "evidence_terms: [<copy the most decision-critical exact terms from MANDATORY_EVIDENCE_TERMS that are actually supported>]\n"
            "reasons: [<evidence-backed bullets; preserve exact skills/tools>]\n"
            "concerns: [<evidence-backed bullets; preserve exact gaps/risks>]\n"
            "interview_plan: [<questions or N/A>]\n"
            "Do not change the schema labels."
        )

    def run_baseline(self, llm: GroqClient, context: str, max_tokens: int) -> LLMResponse:
        return llm.generate(self.system_prompt(), self.user_prompt(context), max_tokens=max_tokens, temperature=0.0)

    def run_optimized(self, llm: GroqClient, context: str, optimizer: AggressiveSafeOptimizer, max_tokens: int) -> tuple[LLMResponse, OptimizationResult]:
        opt = optimizer.optimize(context, task=self.task)
        resp = llm.generate(self.system_prompt(), self.user_prompt(opt.optimized_text), max_tokens=max_tokens, temperature=0.0)
        return resp, opt


def default_agents() -> list[ProdSyncAgent]:
    return [
        ProdSyncAgent("Resume Intelligence Agent", "Extract candidate strengths, weaknesses, skills, experience, project quality, and resume risks."),
        ProdSyncAgent("JD Matching Agent", "Compare candidate evidence against ML Engineer job requirements. Identify match score, missing skills, and hard blockers."),
        ProdSyncAgent("Interview Intelligence Agent", "Evaluate candidate interview answers for technical depth, communication, honesty, and readiness."),
        ProdSyncAgent("Recruiter Decision Agent", "Make final shortlist decision using all previous agent evidence. Return final_score, shortlist_decision, evidence_terms, reasons, concerns, interview_plan."),
    ]
