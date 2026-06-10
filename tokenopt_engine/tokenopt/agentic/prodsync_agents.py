import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..core.optimizer import TokenOptimizer
from ..core.schemas import TokenOptimizationConfig
from ..integrations.groq_client import GroqClient
from ..tokenization import count_tokens
from .compact_memory import build_prodsync_compact_memory, memory_stats, route_context_for_agent
from .rate_limiter import GroqFreeTierPacer


@dataclass
class AgentCallRecord:
    agent_name: str
    objective: str
    agent_mode: str
    estimated_total_tokens_for_pacing: int
    pacer_sleep_seconds: float
    optimizer_metrics: Dict[str, object]
    groq_usage: Dict[str, object]
    estimated_prompt_tokens_sent: int
    actual_prompt_tokens: int
    actual_completion_tokens: int
    actual_total_tokens: int
    prompt_token_estimation_error_pct: float
    actual_cost_usd: float
    reconciled_savings_usd: float
    response_preview: str
    status: str = "ok"
    error: Optional[str] = None


@dataclass
class AgenticRunReport:
    run_id: str
    model: str
    default_mode: str
    started_at_utc: str
    finished_at_utc: str
    total_agents: int
    total_actual_prompt_tokens: int
    total_actual_completion_tokens: int
    total_actual_tokens: int
    total_actual_cost_usd: float
    total_reconciled_savings_usd: float
    total_framework_tokens_saved: int
    average_retention_score: float
    worst_retention_score: float
    accepted_agents: int
    rolled_back_agents: int
    high_risk_agents: int
    total_sleep_seconds: float
    memory_compaction: Dict[str, object]
    calls: List[AgentCallRecord] = field(default_factory=list)
    final_answer: str = ""

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["calls"] = [asdict(c) for c in self.calls]
        return data


AGENT_POLICIES = {
    "ResumeIntelligenceAgent": {
        "mode": "balanced",
        "min_retention": 0.86,
        "max_tokens": 3200,
        "max_completion": 420,
    },
    "JDRequirementAgent": {
        "mode": "balanced",
        "min_retention": 0.86,
        "max_tokens": 3000,
        "max_completion": 380,
    },
    "SemanticFitScoringAgent": {
        "mode": "conservative",
        "min_retention": 0.90,
        "max_tokens": 3000,
        "max_completion": 420,
    },
    "SkillGapReadinessAgent": {
        "mode": "balanced",
        "min_retention": 0.86,
        "max_tokens": 2800,
        "max_completion": 500,
    },
    "InterviewQuestionAgent": {
        "mode": "aggressive",
        "min_retention": 0.82,
        "max_tokens": 2400,
        "max_completion": 520,
    },
    "RecruiterDecisionAgent": {
        "mode": "conservative",
        "min_retention": 0.90,
        "max_tokens": 2800,
        "max_completion": 420,
    },
}


class ProdSyncAgenticGroqTester:
    """ProdSync-style multi-agent test harness for TokenOpt + Groq.

    v1.3 improvements:
    - Agent-specific optimization modes and thresholds.
    - Deterministic compact memory routing to avoid repeating full resume/JD.
    - Smarter retention scoring from the framework.
    - Actual Groq usage reconciliation for prompt-token estimation error and savings.
    """

    def __init__(
        self,
        groq_client: Optional[GroqClient] = None,
        pacer: Optional[GroqFreeTierPacer] = None,
        default_mode: str = None,
        max_completion_tokens: int = 450,
        temperature: float = 0.15,
        input_price_per_million_tokens: float = 0.59,
        output_price_per_million_tokens: float = 0.79,
    ):
        self.default_mode = default_mode or os.getenv("TOKENOPT_MODE", "balanced")
        self.client = groq_client or GroqClient()
        self.pacer = pacer or GroqFreeTierPacer(
            tokens_per_minute=int(os.getenv("GROQ_SAFE_TPM", "9000")),
            requests_per_minute=int(os.getenv("GROQ_SAFE_RPM", "20")),
            min_delay_seconds=float(os.getenv("GROQ_REQUEST_GAP_SECONDS", "8")),
        )
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.input_price = input_price_per_million_tokens
        self.output_price = output_price_per_million_tokens
        self.agent_policies = dict(AGENT_POLICIES)

    def _optimizer_for_agent(self, agent_name: str) -> TokenOptimizer:
        policy = self.agent_policies.get(agent_name, {})
        mode = os.getenv(f"TOKENOPT_{agent_name.upper()}_MODE", policy.get("mode", self.default_mode))
        min_retention = float(os.getenv(f"TOKENOPT_{agent_name.upper()}_RETENTION", str(policy.get("min_retention", 0.86))))
        max_tokens = int(os.getenv(f"TOKENOPT_{agent_name.upper()}_MAX_INPUT", str(policy.get("max_tokens", 3000))))
        max_completion = int(policy.get("max_completion", self.max_completion_tokens))
        return TokenOptimizer(
            TokenOptimizationConfig(
                mode=mode,
                max_input_tokens=max_tokens,
                min_retention_score=min_retention,
                expected_output_tokens=max_completion,
                input_price_per_million_tokens=self.input_price,
                output_price_per_million_tokens=self.output_price,
                debug=True,
            )
        )

    def _actual_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return round(
            (prompt_tokens / 1_000_000.0) * self.input_price
            + (completion_tokens / 1_000_000.0) * self.output_price,
            8,
        )

    def _reconciled_savings(self, estimated_savings: float, estimated_prompt_sent: int, actual_prompt_tokens: int) -> float:
        if estimated_prompt_sent <= 0:
            return round(float(estimated_savings or 0.0), 8)
        ratio = actual_prompt_tokens / max(1, estimated_prompt_sent)
        return round(max(0.0, float(estimated_savings or 0.0) * ratio), 8)

    def _system_prompt(self, agent_name: str) -> str:
        return (
            f"You are {agent_name}, a precise ProdSync AI agent. "
            "Use only the supplied context. Preserve constraints, numbers, skills, dates, named entities, and role requirements. "
            "Return compact JSON-like structured text. Do not hallucinate. "
            "If evidence is missing, say missing_evidence instead of inventing."
        )

    def _call_agent(self, agent_name: str, objective: str, prompt: str, context: List[str], query: str) -> AgentCallRecord:
        optimizer = self._optimizer_for_agent(agent_name)
        policy = self.agent_policies.get(agent_name, {})
        max_completion = int(policy.get("max_completion", self.max_completion_tokens))

        optimized = optimizer.optimize(prompt=prompt, context=context, query=query)
        estimated_prompt_sent = count_tokens(optimized.optimized_prompt) + count_tokens(self._system_prompt(agent_name))
        estimated_total = estimated_prompt_sent + max_completion + 120
        pacer_state = self.pacer.wait(estimated_total)

        try:
            response = self.client.chat(
                prompt=optimized.optimized_prompt,
                system_prompt=self._system_prompt(agent_name),
                temperature=self.temperature,
                max_tokens=max_completion,
            )
            usage = response.get("usage", {}) or {}
            actual_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            actual_completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            actual_total_tokens = int(usage.get("total_tokens", actual_prompt_tokens + actual_completion_tokens) or 0)
            actual_cost = self._actual_cost(actual_prompt_tokens, actual_completion_tokens)
            estimation_error_pct = round(((actual_prompt_tokens - estimated_prompt_sent) / max(1, estimated_prompt_sent)) * 100, 4)
            reconciled_savings = self._reconciled_savings(
                float(optimized.metrics.estimated_savings or 0.0),
                estimated_prompt_sent,
                actual_prompt_tokens,
            )
            content = str(response.get("content", ""))
            return AgentCallRecord(
                agent_name=agent_name,
                objective=objective,
                agent_mode=optimizer.config.mode,
                estimated_total_tokens_for_pacing=estimated_total,
                pacer_sleep_seconds=pacer_state.last_sleep_seconds,
                optimizer_metrics=optimized.metrics.to_dict(),
                groq_usage=usage,
                estimated_prompt_tokens_sent=estimated_prompt_sent,
                actual_prompt_tokens=actual_prompt_tokens,
                actual_completion_tokens=actual_completion_tokens,
                actual_total_tokens=actual_total_tokens,
                prompt_token_estimation_error_pct=estimation_error_pct,
                actual_cost_usd=actual_cost,
                reconciled_savings_usd=reconciled_savings,
                response_preview=content[:1200],
            )
        except Exception as exc:
            return AgentCallRecord(
                agent_name=agent_name,
                objective=objective,
                agent_mode=optimizer.config.mode,
                estimated_total_tokens_for_pacing=estimated_total,
                pacer_sleep_seconds=pacer_state.last_sleep_seconds,
                optimizer_metrics=optimized.metrics.to_dict(),
                groq_usage={},
                estimated_prompt_tokens_sent=estimated_prompt_sent,
                actual_prompt_tokens=0,
                actual_completion_tokens=0,
                actual_total_tokens=0,
                prompt_token_estimation_error_pct=0.0,
                actual_cost_usd=0.0,
                reconciled_savings_usd=0.0,
                response_preview="",
                status="error",
                error=str(exc),
            )

    def run(self, candidate_profile: str, job_description: str, evidence_context: List[str]) -> AgenticRunReport:
        started = datetime.now(timezone.utc)
        run_id = started.strftime("prodsync-agentic-v13-%Y%m%d-%H%M%S")

        memory = build_prodsync_compact_memory(candidate_profile, job_description, evidence_context)
        mem_stats = memory_stats(memory, candidate_profile, job_description, evidence_context)
        calls: List[AgentCallRecord] = []
        previous_outputs: List[str] = []

        agents = [
            (
                "ResumeIntelligenceAgent",
                "Extract candidate skills, experience, project evidence, and risk signals.",
                "Analyze the candidate profile for ProdSync. Extract JSON with skills, years_of_experience, project_evidence, seniority, strengths, weak_evidence, missing_evidence, and red_flags.",
                "candidate skills experience projects evidence red flags",
            ),
            (
                "JDRequirementAgent",
                "Extract mandatory and preferred role requirements from the JD.",
                "Analyze the job description. Return JSON with must_have_requirements, good_to_have_skills, domain_expectations, screening_constraints, and scoring_weights.",
                "job description mandatory requirements preferred skills constraints",
            ),
            (
                "SemanticFitScoringAgent",
                "Score candidate-role fit with explainable reasoning.",
                "Compare candidate evidence against JD requirements. Return fit_score out of 100, matched_skills, partial_matches, missing_skills, evidence_strength, rejection_risks, and scoring_rationale.",
                "candidate role fit score matched missing skills evidence",
            ),
            (
                "SkillGapReadinessAgent",
                "Create a 2-8 week readiness plan without losing constraints.",
                "Create a practical readiness plan. Include priority_gaps, week_by_week_plan, project_improvements, resume_fixes, and interview_readiness_actions.",
                "skill gap readiness plan weekly project resume interview",
            ),
            (
                "InterviewQuestionAgent",
                "Generate targeted AI mock interview questions.",
                "Generate targeted mock interview questions: technical, project deep-dive, system design, ML fundamentals, behavioral, and recruiter screening. Avoid repeating generic questions.",
                "mock interview questions technical project system design behavioral",
            ),
            (
                "RecruiterDecisionAgent",
                "Produce final recruiter-facing recommendation.",
                "Produce a recruiter-facing JSON recommendation: shortlist_decision, hireability_label, risks, reasons, interview_focus_areas, next_step, fit_score, and missing_evidence.",
                "recruiter recommendation shortlist hireability risks next steps",
            ),
        ]

        for agent_name, objective, prompt, query in agents:
            routed_context = route_context_for_agent(
                agent_name,
                memory,
                candidate_profile,
                job_description,
                evidence_context,
                previous_outputs,
            )
            record = self._call_agent(agent_name, objective, prompt, routed_context, query)
            calls.append(record)
            if record.response_preview:
                # Compact chain memory: never append raw long output blindly.
                previous_outputs.append(f"[{agent_name}_OUTPUT]\n{record.response_preview[:1100]}")

        finished = datetime.now(timezone.utc)
        retentions = [float(c.optimizer_metrics.get("retention_score", 0.0) or 0.0) for c in calls]
        tokens_saved = [int(c.optimizer_metrics.get("tokens_saved", 0) or 0) for c in calls]
        accepted = sum(1 for c in calls if c.optimizer_metrics.get("status") == "accepted")
        rolled = sum(1 for c in calls if c.optimizer_metrics.get("status") == "rolled_back")
        high_risk = sum(1 for c in calls if c.optimizer_metrics.get("risk_label") == "HIGH")

        final_answer = calls[-1].response_preview if calls else ""
        return AgenticRunReport(
            run_id=run_id,
            model=self.client.model,
            default_mode=self.default_mode,
            started_at_utc=started.isoformat(),
            finished_at_utc=finished.isoformat(),
            total_agents=len(calls),
            total_actual_prompt_tokens=sum(c.actual_prompt_tokens for c in calls),
            total_actual_completion_tokens=sum(c.actual_completion_tokens for c in calls),
            total_actual_tokens=sum(c.actual_total_tokens for c in calls),
            total_actual_cost_usd=round(sum(c.actual_cost_usd for c in calls), 8),
            total_reconciled_savings_usd=round(sum(c.reconciled_savings_usd for c in calls), 8),
            total_framework_tokens_saved=sum(tokens_saved),
            average_retention_score=round(sum(retentions) / max(1, len(retentions)), 4),
            worst_retention_score=round(min(retentions) if retentions else 0.0, 4),
            accepted_agents=accepted,
            rolled_back_agents=rolled,
            high_risk_agents=high_risk,
            total_sleep_seconds=round(self.pacer.total_sleep_seconds, 3),
            memory_compaction=mem_stats,
            calls=calls,
            final_answer=final_answer,
        )


def save_agentic_report(report: AgenticRunReport, output_dir: str = "reports") -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{report.run_id}.json")
    csv_path = os.path.join(output_dir, f"{report.run_id}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    headers = [
        "agent_name", "status", "agent_mode", "risk_label", "pacer_sleep_seconds",
        "estimated_total_tokens_for_pacing", "estimated_prompt_tokens_sent", "actual_prompt_tokens",
        "actual_completion_tokens", "actual_total_tokens", "prompt_token_estimation_error_pct",
        "actual_cost_usd", "reconciled_savings_usd", "original_tokens", "optimized_tokens",
        "tokens_saved", "reduction_percentage", "retention_score", "entity_retention",
        "numeric_retention", "constraint_retention", "semantic_similarity", "keyword_coverage",
        "optimizer_status", "optimizer_reason", "error"
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for c in report.calls:
            m = c.optimizer_metrics
            row = [
                c.agent_name,
                c.status,
                c.agent_mode,
                str(m.get("risk_label", "")),
                str(c.pacer_sleep_seconds),
                str(c.estimated_total_tokens_for_pacing),
                str(c.estimated_prompt_tokens_sent),
                str(c.actual_prompt_tokens),
                str(c.actual_completion_tokens),
                str(c.actual_total_tokens),
                str(c.prompt_token_estimation_error_pct),
                str(c.actual_cost_usd),
                str(c.reconciled_savings_usd),
                str(m.get("original_tokens", "")),
                str(m.get("optimized_tokens", "")),
                str(m.get("tokens_saved", "")),
                str(m.get("reduction_percentage", "")),
                str(m.get("retention_score", "")),
                str(m.get("entity_retention", "")),
                str(m.get("numeric_retention", "")),
                str(m.get("constraint_retention", "")),
                str(m.get("semantic_similarity", "")),
                str(m.get("keyword_coverage", "")),
                str(m.get("status", "")),
                str(m.get("reason", "") or ""),
                str(c.error or "").replace(",", ";"),
            ]
            f.write(",".join(v.replace("\n", " ").replace("\r", " ") for v in row) + "\n")
    return {"json": json_path, "csv": csv_path}
