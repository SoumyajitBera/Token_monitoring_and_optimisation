import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenopt.agentic import ProdSyncAgenticGroqTester, save_agentic_report


def load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def repeated_policy_context() -> list[str]:
    base = [
        "ProdSync evaluates employability readiness for Tier-2 and Tier-3 candidates using resume evidence, GitHub evidence, JD matching, skill-gap planning, and AI mock interview readiness.",
        "Mandatory safety rule: Do not recommend shortlist when mandatory JD constraints are missing unless the missing requirement is explicitly marked trainable within 2 weeks.",
        "Scoring rule: Technical fit is weighted 40%, project evidence 25%, communication readiness 15%, role alignment 10%, and learning velocity 10%.",
        "Sensitive constraint: Never inflate experience years. If the candidate has 2.9 years, preserve 2.9 years and do not round it to 3+ years.",
        "Sensitive constraint: Preserve exact model and tool names such as FastAPI, LangChain, Milvus, Groq, Watsonx, SentenceTransformers, scikit-learn, PyTorch, Docker, GCP Cloud Run, and TiDB.",
        "Recruiter output must include: shortlist decision, fit score, missing evidence, risks, interview focus areas, and final next action.",
        "Candidate readiness output must include a 2-8 week plan with week-wise actions, project improvements, resume fixes, and mock interview practice.",
        "Duplicate note: ProdSync evaluates employability readiness for Tier-2 and Tier-3 candidates using resume evidence, GitHub evidence, JD matching, skill-gap planning, and AI mock interview readiness.",
        "Duplicate note: Recruiter output must include shortlist decision, fit score, missing evidence, risks, interview focus areas, and final next action.",
    ]
    return base * 4


CANDIDATE_PROFILE = """
Candidate: Soumyajit Bera
Current role: AI Engineer at IBM
Experience: 2.9 years at IBM plus 1 year Nokia internship
Core skills: Python, FastAPI, SQL, Pandas, scikit-learn, TensorFlow, PyTorch, LangChain, Milvus, SentenceTransformers, Watsonx, Groq API integration, Docker, GCP Cloud Run, TiDB, Elasticsearch, spaCy, KeyBERT
Projects:
1. ProdSync AI SaaS: AI-powered employability and hiring intelligence platform for Tier-2/Tier-3 talent ecosystem. Includes ATS resume parsing, AI role recommendation, JD-resume matching, skill-gap plan, recruiter shortlisting, and AI mock interviews.
2. Natural Language to SQL asset: role-based SQL generation with deterministic governance metrics including schema grounding, SQL safety, executability, intent coverage, result shape validation, and numeric faithfulness.
3. RAG pipeline: Milvus vector store, SentenceTransformers embeddings, Watsonx answering, document ingestion, metadata extraction, retrieval API, and PDF generation.
4. Token monitoring and optimisation: mathematical token optimization layer using retention validation, semantic deduplication, cost tracking, and rollback.
5. GCP deployment: deployed backend and frontend services using Cloud Run, Artifact Registry, Secret Manager, and CI/CD.
Strengths: AI engineering, backend API integration, RAG, LLM apps, governance metrics, practical deployment.
Weaknesses / missing evidence: limited formal MLOps production ownership, no direct large-scale Kubernetes ownership, limited paid SaaS customer traction yet, startup is MVP stage.
Location preference: Kolkata preferred, but open to Bangalore/remote depending on offer.
Expected CTC: 25 LPA negotiable.
"""

JOB_DESCRIPTION = """
Role: AI/ML Engineer - Agentic AI and LLM Applications
Mandatory requirements:
- 2+ years professional AI/ML or LLM application development experience.
- Strong Python and backend API development experience.
- Experience with RAG, vector databases, embeddings, and LLM integration.
- Ability to build reliable evaluation metrics and guardrails.
- Experience deploying services on cloud platforms.
Preferred requirements:
- LangChain or agentic workflow experience.
- FastAPI, Docker, CI/CD, and cloud deployment exposure.
- Experience with cost optimization and token usage monitoring.
- Strong communication and product thinking.
Do not shortlist if the candidate lacks Python, API development, and LLM integration evidence.
Interview should test: RAG design, LLM evaluation, prompt optimization, vector search, deployment, cost monitoring, and production safety.
"""


def main() -> None:
    load_dotenv()

    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY missing. Copy .env.example to .env and fill your key.")

    tester = ProdSyncAgenticGroqTester()
    report = tester.run(
        candidate_profile=CANDIDATE_PROFILE,
        job_description=JOB_DESCRIPTION,
        evidence_context=repeated_policy_context(),
    )
    paths = save_agentic_report(report)

    print("\n=== PRODSYNC AGENTIC GROQ TEST SUMMARY V1.3 ===")
    print(json.dumps({
        "run_id": report.run_id,
        "model": report.model,
        "default_mode": report.default_mode,
        "total_agents": report.total_agents,
        "accepted_agents": report.accepted_agents,
        "rolled_back_agents": report.rolled_back_agents,
        "high_risk_agents": report.high_risk_agents,
        "total_actual_prompt_tokens": report.total_actual_prompt_tokens,
        "total_actual_completion_tokens": report.total_actual_completion_tokens,
        "total_actual_tokens": report.total_actual_tokens,
        "total_actual_cost_usd": report.total_actual_cost_usd,
        "total_reconciled_savings_usd": report.total_reconciled_savings_usd,
        "total_framework_tokens_saved": report.total_framework_tokens_saved,
        "average_retention_score": report.average_retention_score,
        "worst_retention_score": report.worst_retention_score,
        "memory_compaction": report.memory_compaction,
        "total_sleep_seconds": report.total_sleep_seconds,
        "report_json": paths["json"],
        "report_csv": paths["csv"],
    }, indent=2))

    print("\n=== AGENT CALL BREAKDOWN ===")
    for c in report.calls:
        m = c.optimizer_metrics
        print(json.dumps({
            "agent": c.agent_name,
            "status": c.status,
            "agent_mode": c.agent_mode,
            "sleep_before_call_seconds": c.pacer_sleep_seconds,
            "actual_total_tokens": c.actual_total_tokens,
            "actual_cost_usd": c.actual_cost_usd,
            "reconciled_savings_usd": c.reconciled_savings_usd,
            "prompt_token_estimation_error_pct": c.prompt_token_estimation_error_pct,
            "optimizer_status": m.get("status"),
            "risk_label": m.get("risk_label"),
            "tokens_saved": m.get("tokens_saved"),
            "reduction_percentage": m.get("reduction_percentage"),
            "retention_score": m.get("retention_score"),
            "constraint_retention": m.get("constraint_retention"),
            "error": c.error,
        }, indent=2))

    print("\n=== FINAL RECRUITER DECISION PREVIEW ===")
    print(report.final_answer)


if __name__ == "__main__":
    main()
