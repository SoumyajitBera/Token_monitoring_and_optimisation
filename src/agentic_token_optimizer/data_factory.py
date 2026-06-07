from __future__ import annotations


def build_context(profile: str = "large") -> str:
    repeat = 1 if profile == "medium" else 2
    resume = """
CANDIDATE_RESUME:
Name: Soumyajit Bera
Target Role: ML Engineer / Applied AI Engineer
Experience: 2.9 years AI Engineer at IBM, 1 year Nokia internship.
Core Skills: Python, FastAPI, SQL, LangChain, RAG, vector databases, Milvus, Docker, GCP Cloud Run, IBM watsonx, ML evaluation, NLP, computer vision, statistical modeling.
Projects:
1. ProdSync AI SaaS: candidate intelligence, ATS resume parsing, JD semantic matching, GitHub profile analysis, recruiter shortlisting, AI mock interview, readiness scoring, skill-gap intelligence, and interview report generation.
2. Role-based NLQ-SQL platform: natural-language-to-SQL generation with role permissions, admin/super-user/normal-user workflows, SQL injection prevention, schema grounding, SQL executability checks, deterministic agent monitoring metrics, and result validation.
3. Token optimizer notebook: token counting, redundancy removal, complexity score, adaptive token budget, cost estimation.
4. RAG asset: Milvus vector DB, SentenceTransformers, watsonx LLM, document ingestion, retrieval, FastAPI service.
Strengths: Applied AI engineering, API integration, RAG, vector DB, production deployment, cost awareness, agentic workflows, strong project ownership.
Concerns: Limited deep MLOps experience with Kubernetes, feature stores, model registry lifecycle, online monitoring, model drift and concept drift in production. Needs stronger quality evaluation using human labels and task-specific acceptance metrics. Some benchmark data is synthetic and requires real-world validation logs.
Constraints: Do not fabricate deep MLOps experience. Do not claim production Kubernetes ownership. Must honestly identify gaps.
"""
    jd = """
JOB_DESCRIPTION:
Role: ML Engineer
Required: Python, FastAPI or backend API development, SQL, cloud deployment, CI/CD, Docker, ML fundamentals, model evaluation, LLM application development, vector databases, RAG, production API ownership.
Preferred: Kubernetes, feature stores, model registry, MLflow, real-time monitoring, A/B testing, model drift detection, data validation, scalable model serving, observability, cross-functional collaboration with product and recruiter-facing teams.
Hard Requirements: Candidate must be able to build agentic AI systems, optimize LLM cost, evaluate model quality, write clean Python services, and explain tradeoffs. Candidate must not exaggerate experience. Candidate must identify gaps honestly.
Hiring Risk Flags: shallow MLOps, no real production monitoring, only notebook-level experiments, no ownership of cloud deployment, no evidence of maintaining APIs in production.
"""
    github = """
GITHUB_AND_PROJECT_EVIDENCE:
Repository evidence shows FastAPI backend work, Cloud Run deployment commands, Dockerfiles, Vite frontend integration, GCP Secret Manager usage, TiDB connectivity setup, CORS debugging, Artifact Registry setup, and CI/CD YAML workflows.
Evidence also shows agentic NLQ-SQL monitoring with deterministic metrics: schema grounding, SQL safety, executability, intent coverage, result shape validation, numeric faithfulness. These are not LLM-as-judge metrics.
Evidence shows token optimization work with baseline vs optimized Groq calls, rate-limit management, output printing, token/cost analytics, quality warnings, and need for real data validation.
Weak evidence for Kubernetes, feature stores, model registry, online monitoring, A/B testing, and model drift lifecycle.
"""
    interview = """
MOCK_INTERVIEW_TRANSCRIPT:
Interviewer: Explain how you would reduce LLM cost in a multi-agent hiring product.
Candidate: I would compress resume, JD, GitHub summaries, and interview history before every agent call. I would keep code blocks, JSON, negative constraints, salary constraints, required skills, do-not conditions, candidate evidence, and recruiter query intent. I would use semantic pruning and adaptive output budgets.
Interviewer: What should not be removed?
Candidate: Functional constraints, salary constraints, required skills, do-not conditions, code, schemas, candidate evidence, recruiter query intent, score evidence, and interview concerns.
Interviewer: Weakness?
Candidate: Need stronger quality evaluation using human labels and task-specific acceptance metrics. I also need deeper MLOps experience.
Interviewer: How would you handle MLOps gap?
Candidate: I would learn model registry, CI/CD for models, drift monitoring, feature stores, Kubernetes deployment, and production observability. I would not falsely claim deep experience today.
Interviewer: Explain model drift and concept drift.
Candidate: Model drift is degradation in model performance over time due to data changes. Concept drift happens when the relationship between input features and target labels changes. In production I would monitor data distribution, prediction distribution, performance metrics, and alert thresholds.
Interviewer: How would you prove your token optimizer does not corrupt hiring decisions?
Candidate: I would compare baseline Groq and optimized Groq outputs on the same candidate/JD data, preserve final decision, compare score deltas, keyword retention, semantic similarity, and human rubric labels.
"""
    noisy = """
NOISY_REPEATED_CONTEXT:
Please kindly note this candidate is applying for ML Engineer. Please kindly note this candidate is applying for ML Engineer.
The recruiter wants honest evaluation. The recruiter wants honest evaluation. The recruiter wants honest evaluation.
Do not remove the weakness about MLOps. Do not remove the weakness about MLOps.
Candidate has Python FastAPI SQL RAG vector DB experience. Candidate has Python FastAPI SQL RAG vector DB experience.
This sentence is filler and does not add meaningful evidence. This sentence is filler and does not add meaningful evidence.
"""
    blocks = [resume, jd, github, interview, noisy]
    return "\n".join(blocks * repeat)
