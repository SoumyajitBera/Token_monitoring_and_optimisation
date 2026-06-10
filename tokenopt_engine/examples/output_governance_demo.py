import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenopt.governance import (
    normalized_output_text,
    schema_instruction_for_contract,
    validate_and_normalize_agent_output,
    validate_evidence_consistency,
)


def main() -> None:
    loose_output = """
    {
      "shortlist_decision": "yes",
      "hireability_label": "strong",
      "risks": "limited Kubernetes evidence; formal MLOps ownership not proven",
      "reasons": ["strong Python", "RAG and LLM integration evidence"],
      "interview_focus_areas": ["RAG design", "deployment", "cost monitoring"],
      "next_step": "schedule interview",
      "fit_score": 0.8,
      "missing_evidence": "none"
    }
    """

    result = validate_and_normalize_agent_output("RecruiterDecisionAgent", loose_output)
    consistency = validate_evidence_consistency(result.normalized_output)

    print("=== OUTPUT CONTRACT INSTRUCTION ===")
    print(schema_instruction_for_contract("RecruiterDecisionAgent"))

    print("\n=== GOVERNANCE RESULT V1.5 ===")
    print(json.dumps({
        "schema_valid": result.schema_valid,
        "schema_repaired": result.repaired,
        "schema_quality_score": result.schema_quality_score,
        "errors": result.errors,
        "warnings": result.warnings,
        "repair_actions": result.repair_actions,
        "evidence_consistent": consistency.consistent,
        "evidence_consistency_score": consistency.score,
        "evidence_warnings": consistency.warnings,
    }, indent=2))

    print("\n=== NORMALIZED OUTPUT ===")
    print(normalized_output_text(result))


if __name__ == "__main__":
    main()
