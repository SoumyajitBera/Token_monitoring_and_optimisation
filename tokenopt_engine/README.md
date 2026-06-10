# TokenOpt Engine v1.5

Dependency-light Python framework for safe LLM token optimization, cost tracking, retention validation, rollback, Groq testing, agentic benchmarking, output governance, and schema-aware optimization.

v1.5 keeps the v1.4 pipeline intact and adds framework-level reliability improvements. The improvements are generic and reusable for any LLM agent workflow, not just ProdSync.

## What v1.5 Adds

- Schema-aware optimization policies.
- Protected schema-critical instructions during prompt optimization.
- Required-field and non-empty-field contract validation.
- Optional schema-repair retry when the first output violates the contract.
- Generic evidence-consistency validation for structured outputs.
- Schema quality score per agent.
- Retry-token and retry-cost tracking.
- Evidence-consistency fields in JSON/CSV reports.

## v1.4 Features Preserved

- Agent output contract validation.
- JSON-like output parsing with safe fallback.
- Per-agent schema normalization.
- Stable enum normalization for decision outputs.
- Fit score normalization to 0-100 scale.
- List normalization for fields such as `risks` and `missing_evidence`.
- Schema validity, repair actions, warnings, and errors in JSON/CSV reports.

## v1.3 Features Preserved

- Smarter constraint retention using logical constraint groups instead of brittle exact keyword matching.
- Agent-specific optimization modes.
- Compact memory routing to avoid repeating full resume/JD/context across every agent.
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

Governance-only demo:

```bash
python examples/output_governance_demo.py
```

## Recommended Free-Tier Pacing

Default values:

```env
GROQ_SAFE_TPM="9000"
GROQ_SAFE_RPM="20"
GROQ_REQUEST_GAP_SECONDS="8"
TOKENOPT_ENABLE_SCHEMA_RETRY="true"
```

If you hit `429`, reduce to:

```env
GROQ_SAFE_TPM="6000"
GROQ_SAFE_RPM="10"
GROQ_REQUEST_GAP_SECONDS="12"
```

If you want to avoid extra retry calls during free-tier experiments:

```env
TOKENOPT_ENABLE_SCHEMA_RETRY="false"
```

## Agent Policy Defaults

The example ProdSync harness uses these defaults, but the framework features are generic.

| Agent | Default Mode | Schema Strict | Why |
|---|---:|---:|---|
| ResumeIntelligenceAgent | balanced | true | Extractive output with required fields |
| JDRequirementAgent | balanced | true | Requirement extraction with constraints |
| SemanticFitScoringAgent | conservative | true | Scoring is sensitive |
| SkillGapReadinessAgent | balanced | true | Planning can be compacted but must keep structure |
| InterviewQuestionAgent | aggressive requested, auto-balanced when schema-strict | true | Schema completeness matters |
| RecruiterDecisionAgent | conservative | true | Final decision is sensitive |

Schema-strict mode is framework-level. If a contract has non-empty required fields, the optimizer protects the contract and automatically avoids unsafe aggressive pruning.

## Public API

```python
from tokenopt import TokenOptimizer
from tokenopt.core.schemas import TokenOptimizationConfig

optimizer = TokenOptimizer(TokenOptimizationConfig(
    mode="balanced",
    min_retention_score=0.88,
))

result = optimizer.optimize(
    prompt="...",
    context=["..."],
    query="...",
)

print(result.optimized_prompt)
print(result.metrics.to_dict())
```

## Schema-Aware Optimization API

```python
from tokenopt import TokenOptimizer
from tokenopt.core.schemas import TokenOptimizationConfig
from tokenopt.governance import schema_instruction_for_contract, required_fields_for_contract, get_output_contract

contract = get_output_contract("RecruiterDecisionAgent")
schema_text = schema_instruction_for_contract("RecruiterDecisionAgent", contract)

optimizer = TokenOptimizer(TokenOptimizationConfig(
    mode="aggressive",
    schema_strict=True,
    protected_texts=[schema_text],
    schema_critical_terms=required_fields_for_contract(contract),
))

result = optimizer.optimize(
    prompt="Produce the decision JSON.",
    context=["candidate and job context..."],
    query="final hiring decision",
    protected_texts=[schema_text],
    schema_critical_terms=required_fields_for_contract(contract),
)
```

## Output Governance API

```python
from tokenopt.governance import validate_and_normalize_agent_output, validate_evidence_consistency

result = validate_and_normalize_agent_output("RecruiterDecisionAgent", raw_llm_output)
consistency = validate_evidence_consistency(result.normalized_output)

print(result.schema_valid)
print(result.schema_quality_score)
print(result.repair_actions)
print(consistency.warnings)
print(result.normalized_output)
```

Example repair:

```text
"fit_score": 0.8             -> 80
"shortlist_decision": "yes"  -> "SHORTLIST"
"missing_evidence": "none"   -> []
"risks": "a; b"              -> ["a", "b"]
```

## Reports

Agentic reports are written to:

```text
reports/<run_id>.json
reports/<run_id>.csv
```

The report includes:

- token usage
- actual Groq usage
- estimated vs actual token error
- cost and reconciled savings
- retention score
- constraint retention
- risk label
- schema validity
- schema repair actions
- schema retry attempts and cost
- evidence consistency score
- memory compaction metrics

## Important Design Note

v1.5 does not hardcode a fix for one agent. It makes the framework aware of structured-output contracts. Any use case can pass a contract, protected schema text, required fields, and non-empty requirements. The optimizer then preserves those instructions during compression, and the governance layer validates the output after generation.
