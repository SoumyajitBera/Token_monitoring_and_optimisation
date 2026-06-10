import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenopt import TokenOptimizer, TokenOptimizationConfig
from tokenopt.governance import get_output_contract, required_fields_for_contract, schema_instruction_for_contract


def main() -> None:
    contract = get_output_contract("InterviewQuestionAgent")
    required_fields = required_fields_for_contract(contract)
    compact_protected = "REQUIRED_JSON_FIELDS: " + ", ".join(required_fields)
    schema_text = schema_instruction_for_contract("InterviewQuestionAgent", contract)

    prompt = "Generate targeted interview questions for the supplied candidate and role. Return JSON."
    context = [
        "Candidate has Python, FastAPI, RAG, vector search, Groq integration, Docker, GCP Cloud Run, and token optimization experience. " * 4,
        "Role requires RAG design, LLM evaluation, prompt optimization, vector search, deployment, cost monitoring, and production safety. " * 4,
        schema_text,
    ]

    optimizer = TokenOptimizer(TokenOptimizationConfig(
        mode="aggressive",
        schema_strict=True,
        protected_texts=[compact_protected],
        schema_critical_terms=required_fields,
        max_input_tokens=900,
        debug=True,
    ))

    result = optimizer.optimize(
        prompt=prompt,
        context=context,
        query="interview question JSON output",
        protected_texts=[compact_protected],
        schema_critical_terms=required_fields,
    )

    print(json.dumps({
        "resolved_mode": optimizer.config.mode,
        "schema_strict": optimizer.config.schema_strict,
        "required_fields": required_fields,
        "metrics": result.metrics.to_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
