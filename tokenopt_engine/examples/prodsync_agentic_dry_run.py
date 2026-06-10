import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenopt.agentic.compact_memory import build_prodsync_compact_memory, memory_stats, route_context_for_agent
from tokenopt.agentic.prodsync_agents import AGENT_POLICIES
from tokenopt.agentic.rate_limiter import GroqFreeTierPacer
from tokenopt.core.optimizer import TokenOptimizer
from tokenopt.core.schemas import TokenOptimizationConfig
from tokenopt.tokenization import count_tokens

from prodsync_agentic_groq_test import CANDIDATE_PROFILE, JOB_DESCRIPTION, repeated_policy_context


def main() -> None:
    pacer = GroqFreeTierPacer(
        tokens_per_minute=int(os.getenv("GROQ_SAFE_TPM", "9000")),
        requests_per_minute=int(os.getenv("GROQ_SAFE_RPM", "20")),
        min_delay_seconds=float(os.getenv("GROQ_REQUEST_GAP_SECONDS", "8")),
        dry_run=True,
    )
    evidence = repeated_policy_context()
    memory = build_prodsync_compact_memory(CANDIDATE_PROFILE, JOB_DESCRIPTION, evidence)
    mem_stats = memory_stats(memory, CANDIDATE_PROFILE, JOB_DESCRIPTION, evidence)

    prompts = [
        ("ResumeIntelligenceAgent", "Extract candidate skills, evidence, and red flags.", "candidate skills experience projects evidence red flags"),
        ("JDRequirementAgent", "Extract mandatory and preferred JD requirements.", "job description mandatory requirements preferred skills constraints"),
        ("SemanticFitScoringAgent", "Compare candidate and JD and score fit.", "candidate role fit score matched missing skills evidence"),
        ("SkillGapReadinessAgent", "Create readiness plan.", "skill gap readiness plan weekly project resume interview"),
        ("InterviewQuestionAgent", "Generate mock interview questions.", "mock interview questions technical project system design behavioral"),
        ("RecruiterDecisionAgent", "Make recruiter decision.", "recruiter recommendation shortlist hireability risks next steps"),
    ]
    rows = []
    previous_outputs = []
    for name, prompt, query in prompts:
        policy = AGENT_POLICIES.get(name, {})
        optimizer = TokenOptimizer(TokenOptimizationConfig(
            mode=policy.get("mode", os.getenv("TOKENOPT_MODE", "balanced")),
            max_input_tokens=int(policy.get("max_tokens", 3000)),
            min_retention_score=float(policy.get("min_retention", 0.86)),
            expected_output_tokens=int(policy.get("max_completion", 450)),
            debug=True,
        ))
        context = route_context_for_agent(name, memory, CANDIDATE_PROFILE, JOB_DESCRIPTION, evidence, previous_outputs)
        result = optimizer.optimize(prompt=prompt, context=context, query=query)
        est_total = count_tokens(result.optimized_prompt) + int(policy.get("max_completion", 450)) + 120
        state = pacer.wait(est_total)
        rows.append({
            "agent": name,
            "mode": optimizer.config.mode,
            "estimated_total_tokens_for_pacing": est_total,
            "dry_run_sleep_seconds": state.last_sleep_seconds,
            "optimizer_status": result.metrics.status,
            "risk_label": result.metrics.risk_label,
            "original_tokens": result.metrics.original_tokens,
            "optimized_tokens": result.metrics.optimized_tokens,
            "tokens_saved": result.metrics.tokens_saved,
            "reduction_percentage": result.metrics.reduction_percentage,
            "retention_score": result.metrics.retention_score,
            "constraint_retention": result.metrics.constraint_retention,
        })
        previous_outputs.append(f"[{name}_SIMULATED_OUTPUT] compact result placeholder for next-agent routing")

    print(json.dumps({
        "note": "Dry run only. No Groq call was made.",
        "safe_tpm": pacer.tokens_per_minute,
        "safe_rpm": pacer.requests_per_minute,
        "request_gap_seconds": pacer.min_delay_seconds,
        "memory_compaction": mem_stats,
        "total_planned_sleep_seconds": pacer.total_sleep_seconds,
        "calls": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
