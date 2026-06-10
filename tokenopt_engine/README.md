# TokenOpt Engine v1.3

Dependency-light Python framework for safe LLM token optimization, cost tracking, retention validation, rollback, Groq testing, and ProdSync-style agentic benchmarking.

## What v1.3 Adds

- Smarter constraint retention using logical constraint groups instead of brittle exact keyword matching.
- Agent-specific optimization modes.
- ProdSync compact memory routing to avoid repeating full resume/JD/context across every agent.
- Actual Groq usage reconciliation: estimated prompt tokens vs real Groq prompt tokens.
- Risk labels: `LOW`, `MEDIUM`, `HIGH`, `ROLLED_BACK`.
- JSON and CSV benchmark reports.
- Free-tier pacing for Groq TPM/RPM safety.

## Install / Run

```bash
cd tokenopt_engine
python examples/prodsync_agentic_dry_run.py
```

For Groq:

```bash
cp .env.example .env
# fill GROQ_API_KEY
python examples/groq_health_check.py
python examples/prodsync_agentic_groq_test.py
```

## Recommended Free-Tier Pacing

Default values:

```env
GROQ_SAFE_TPM="9000"
GROQ_SAFE_RPM="20"
GROQ_REQUEST_GAP_SECONDS="8"
```

If you hit `429`, reduce to:

```env
GROQ_SAFE_TPM="6000"
GROQ_SAFE_RPM="10"
GROQ_REQUEST_GAP_SECONDS="12"
```

## Agent Policy Defaults

| Agent | Mode | Min Retention | Why |
|---|---:|---:|---|
| ResumeIntelligenceAgent | balanced | 0.86 | Extractive but not final decision |
| JDRequirementAgent | balanced | 0.86 | Requirement extraction |
| SemanticFitScoringAgent | conservative | 0.90 | Scoring is sensitive |
| SkillGapReadinessAgent | balanced | 0.86 | Planning can be compacted |
| InterviewQuestionAgent | aggressive | 0.82 | Most compressible |
| RecruiterDecisionAgent | conservative | 0.90 | Final decision is sensitive |

## Public API

```python
from tokenopt import TokenOptimizer
from tokenopt.core.schemas import TokenOptimizationConfig

optimizer = TokenOptimizer(TokenOptimizationConfig(mode="balanced", min_retention_score=0.88))
result = optimizer.optimize(prompt="...", context=["..."], query="...")
print(result.optimized_prompt)
print(result.metrics.to_dict())
```

## Reports

Agentic reports are written to:

```text
reports/<run_id>.json
reports/<run_id>.csv
```

The report includes:

- per-agent optimizer status
- risk label
- actual Groq prompt/completion tokens
- prompt token estimation error
- actual cost
- reconciled savings
- retention dimensions
- memory compaction savings

## Important

This is not a blind prompt compressor. It is a governance layer. If retention is unsafe, rollback returns the original input.
