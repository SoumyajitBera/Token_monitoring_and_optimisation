"""Reusable agent output governance utilities.

v1.5 moves the framework from "repair after the fact" toward
schema-aware execution:

- output contracts are generic dictionaries, not tied to one provider;
- contract instructions can be injected into any LLM system prompt;
- required fields can be marked as non-empty;
- outputs are parsed, normalized, repaired when safe, and audited;
- schema validity is strict: inserted missing required fields are still
  reported as schema errors unless a retry produces the field.

No external dependencies are used.
"""

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GovernanceResult:
    agent_name: str
    raw_output: str
    parsed_output: Dict[str, Any]
    normalized_output: Dict[str, Any]
    schema_valid: bool
    repaired: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    repair_actions: List[str] = field(default_factory=list)
    contract_name: Optional[str] = None
    schema_quality_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Generic contract format:
# {
#   "required": {"field": "list|string|number|dict|score_100|enum_x"},
#   "aliases": {"alternative": "field"},
#   "non_empty": ["field_a", "field_b"],
#   "description": "optional human-readable contract purpose"
# }
AGENT_OUTPUT_CONTRACTS: Dict[str, Dict[str, Any]] = {
    "ResumeIntelligenceAgent": {
        "description": "Candidate evidence extraction output.",
        "required": {
            "skills": "list",
            "years_of_experience": "number",
            "project_evidence": "list",
            "strengths": "list",
            "weak_evidence": "list",
            "missing_evidence": "list",
            "red_flags": "list",
        },
        "non_empty": ["skills", "project_evidence", "strengths"],
        "aliases": {
            "core_skills": "skills",
            "technical_skills": "skills",
            "experience_years": "years_of_experience",
            "projects": "project_evidence",
            "risks": "red_flags",
        },
    },
    "JDRequirementAgent": {
        "description": "Job requirement extraction output.",
        "required": {
            "must_have_requirements": "list",
            "good_to_have_skills": "list",
            "domain_expectations": "list",
            "screening_constraints": "list",
            "scoring_weights": "dict",
        },
        "non_empty": ["must_have_requirements", "screening_constraints"],
        "aliases": {
            "mandatory_requirements": "must_have_requirements",
            "mandatory_skills": "must_have_requirements",
            "required_skills": "must_have_requirements",
            "preferred_requirements": "good_to_have_skills",
            "preferred_skills": "good_to_have_skills",
            "constraints": "screening_constraints",
            "weights": "scoring_weights",
        },
    },
    "SemanticFitScoringAgent": {
        "description": "Explainable fit scoring output.",
        "required": {
            "fit_score": "score_100",
            "matched_skills": "list",
            "partial_matches": "list",
            "missing_skills": "list",
            "evidence_strength": "string",
            "rejection_risks": "list",
            "scoring_rationale": "list",
        },
        "non_empty": ["matched_skills", "scoring_rationale", "evidence_strength"],
        "aliases": {
            "score": "fit_score",
            "role_fit_score": "fit_score",
            "risks": "rejection_risks",
            "rationale": "scoring_rationale",
            "reasoning": "scoring_rationale",
        },
    },
    "SkillGapReadinessAgent": {
        "description": "Readiness and gap remediation plan output.",
        "required": {
            "priority_gaps": "list",
            "week_by_week_plan": "list",
            "project_improvements": "list",
            "resume_fixes": "list",
            "interview_readiness_actions": "list",
        },
        "non_empty": ["priority_gaps", "week_by_week_plan", "interview_readiness_actions"],
        "aliases": {
            "skill_gaps": "priority_gaps",
            "weekly_plan": "week_by_week_plan",
            "plan": "week_by_week_plan",
            "resume_improvements": "resume_fixes",
            "interview_actions": "interview_readiness_actions",
        },
    },
    "InterviewQuestionAgent": {
        "description": "Structured interview question bank output.",
        "required": {
            "technical_questions": "list",
            "project_deep_dive_questions": "list",
            "system_design_questions": "list",
            "ml_fundamentals_questions": "list",
            "behavioral_questions": "list",
            "recruiter_screening_questions": "list",
        },
        "non_empty": [
            "technical_questions",
            "project_deep_dive_questions",
            "system_design_questions",
            "ml_fundamentals_questions",
            "behavioral_questions",
            "recruiter_screening_questions",
        ],
        "aliases": {
            "technical": "technical_questions",
            "technical_question": "technical_questions",
            "project_deep_dive": "project_deep_dive_questions",
            "project_questions": "project_deep_dive_questions",
            "system_design": "system_design_questions",
            "ml_fundamentals": "ml_fundamentals_questions",
            "machine_learning_questions": "ml_fundamentals_questions",
            "behavioral": "behavioral_questions",
            "recruiter_screening": "recruiter_screening_questions",
        },
    },
    "RecruiterDecisionAgent": {
        "description": "Final decision output.",
        "required": {
            "shortlist_decision": "enum_shortlist",
            "hireability_label": "enum_hireability",
            "risks": "list",
            "reasons": "list",
            "interview_focus_areas": "list",
            "next_step": "enum_next_step",
            "fit_score": "score_100",
            "missing_evidence": "list",
        },
        "non_empty": ["shortlist_decision", "hireability_label", "reasons", "interview_focus_areas", "next_step"],
        "aliases": {
            "decision": "shortlist_decision",
            "recommendation": "shortlist_decision",
            "hireability": "hireability_label",
            "focus_areas": "interview_focus_areas",
            "next_action": "next_step",
            "score": "fit_score",
        },
    },
}


ENUMS = {
    "shortlist_decision": {
        "SHORTLIST": {"shortlist", "yes", "recommend", "recommended", "strong yes", "pass"},
        "REJECT": {"reject", "no", "do not shortlist", "not recommended", "fail"},
        "HOLD": {"hold", "maybe", "borderline", "needs review", "manual review"},
    },
    "hireability_label": {
        "HIGHLY_HIREABLE": {"highly hireable", "very strong", "excellent", "strong hire", "high"},
        "STRONG": {"strong", "good", "hireable", "recommended"},
        "MODERATE": {"moderate", "average", "medium", "borderline"},
        "WEAK": {"weak", "low", "not hireable", "poor"},
    },
    "next_step": {
        "SCHEDULE_TECHNICAL_INTERVIEW": {"schedule interview", "schedule technical interview", "technical interview", "interview"},
        "REQUEST_MORE_EVIDENCE": {"request more evidence", "ask for evidence", "need evidence"},
        "REJECT_CANDIDATE": {"reject", "reject candidate", "no next step"},
        "MANUAL_REVIEW": {"manual review", "review", "needs review"},
    },
}


def get_output_contract(name: str) -> Dict[str, Any]:
    """Return a copy of a named output contract."""
    return json.loads(json.dumps(AGENT_OUTPUT_CONTRACTS.get(name, {"required": {}, "aliases": {}, "non_empty": []})))


def required_fields_for_contract(contract: Dict[str, Any]) -> List[str]:
    return list((contract or {}).get("required", {}).keys())


def contract_is_schema_critical(contract: Dict[str, Any]) -> bool:
    if not contract:
        return False
    required = contract.get("required", {}) or {}
    non_empty = contract.get("non_empty", []) or []
    return bool(required and non_empty)


def schema_instruction_for_contract(name: str, contract: Optional[Dict[str, Any]] = None) -> str:
    """Generate provider-independent JSON-output instructions.

    This text is deliberately compact but explicit. It can be appended to a
    system prompt or passed as protected text to the optimizer.
    """
    contract = contract or get_output_contract(name)
    required = contract.get("required", {}) or {}
    if not required:
        return ""
    non_empty = set(contract.get("non_empty", []) or [])
    lines = [
        "OUTPUT CONTRACT - return valid JSON only; no markdown; no prose outside JSON.",
        "Required fields and types:",
    ]
    for field, typ in required.items():
        flag = " non-empty" if field in non_empty else ""
        lines.append(f"- {field}: {typ}{flag}")
    lines.append("Do not omit required fields. Use [] only when evidence is genuinely absent and the field is not marked non-empty.")
    lines.append("Use stable field names exactly as listed above.")
    return "\n".join(lines)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_candidate(text: str) -> str:
    text = _strip_code_fences(text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def parse_json_like_output(text: str) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    candidate = _extract_json_candidate(text or "")
    if not candidate:
        return {}, ["empty_output"]
    try:
        value = json.loads(candidate)
        if isinstance(value, dict):
            return value, warnings
        return {"value": value}, ["json_root_not_object"]
    except Exception as exc:
        warnings.append(f"json_parse_failed:{type(exc).__name__}")
    try:
        value = ast.literal_eval(candidate)
        if isinstance(value, dict):
            warnings.append("parsed_with_ast_literal_eval")
            return value, warnings
    except Exception as exc:
        warnings.append(f"literal_eval_failed:{type(exc).__name__}")
    lines = [ln.strip(" -•\t") for ln in (text or "").splitlines() if ln.strip()]
    if lines:
        warnings.append("fallback_line_extraction_used")
        return {"items": lines}, warnings
    return {}, warnings + ["unable_to_parse_output"]


def _canonical_key(key: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(key).strip().lower()).strip("_")


def _apply_aliases(data: Dict[str, Any], aliases: Dict[str, str], actions: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    alias_map = {_canonical_key(k): v for k, v in (aliases or {}).items()}
    for key, value in (data or {}).items():
        ck = _canonical_key(key)
        target = alias_map.get(ck, ck)
        if target != ck:
            actions.append(f"alias:{key}->{target}")
        if target not in out:
            out[target] = value
        else:
            out[target] = _merge_values(out[target], value)
    return out


def _merge_values(a: Any, b: Any) -> Any:
    if isinstance(a, list):
        if isinstance(b, list):
            return a + [x for x in b if x not in a]
        if b not in a:
            return a + [b]
        return a
    if isinstance(b, list):
        return [a] + [x for x in b if x != a]
    return a if a else b


def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() in {"none", "n/a", "na", "not applicable", "no", "nil", "[]"}:
            return []
        if ";" in s:
            return [x.strip() for x in s.split(";") if x.strip()]
        if "\n" in s:
            return [x.strip(" -•\t") for x in s.splitlines() if x.strip(" -•\t")]
        return [s]
    return [value]


def _to_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "; ".join(str(x) for x in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _to_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _normalize_score_100(value: Any, actions: List[str]) -> int:
    num = _to_number(value)
    if num is None:
        actions.append("default_fit_score:0")
        return 0
    if 0 <= num <= 1:
        actions.append(f"scale_score_0_1_to_100:{num}")
        num *= 100
    num = max(0, min(100, num))
    return int(round(num))


def _normalize_enum(field: str, value: Any, actions: List[str]) -> str:
    raw = _to_string(value).lower().strip()
    enum_map = ENUMS.get(field, {})
    for canonical, variants in enum_map.items():
        if raw == canonical.lower() or raw in variants:
            if raw != canonical.lower():
                actions.append(f"normalize_enum:{field}:{value}->{canonical}")
            return canonical
    for canonical, variants in enum_map.items():
        if any(v in raw for v in variants if len(v) >= 4):
            actions.append(f"normalize_enum_fuzzy:{field}:{value}->{canonical}")
            return canonical
    defaults = {"shortlist_decision": "HOLD", "hireability_label": "MODERATE", "next_step": "MANUAL_REVIEW"}
    actions.append(f"default_enum:{field}->{defaults.get(field, '')}")
    return defaults.get(field, raw.upper() if raw else "UNKNOWN")


def _normalize_field(field: str, expected: str, value: Any, actions: List[str]) -> Any:
    if expected == "list":
        normalized = _to_list(value)
        if not isinstance(value, list):
            actions.append(f"coerce_list:{field}")
        return normalized
    if expected == "dict":
        if isinstance(value, dict):
            return value
        actions.append(f"coerce_dict:{field}")
        return {}
    if expected == "string":
        normalized = _to_string(value)
        if not isinstance(value, str):
            actions.append(f"coerce_string:{field}")
        return normalized
    if expected == "number":
        num = _to_number(value)
        if num is None:
            actions.append(f"default_number:{field}:0")
            return 0.0
        if not isinstance(value, (int, float)):
            actions.append(f"coerce_number:{field}")
        return round(num, 4)
    if expected == "score_100":
        return _normalize_score_100(value, actions)
    if expected == "enum_shortlist":
        return _normalize_enum("shortlist_decision", value, actions)
    if expected == "enum_hireability":
        return _normalize_enum("hireability_label", value, actions)
    if expected == "enum_next_step":
        return _normalize_enum("next_step", value, actions)
    return value


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _schema_quality(required: Dict[str, str], normalized: Dict[str, Any], errors: List[str]) -> float:
    if not required:
        return 1.0
    present = 0
    non_empty = 0
    for field in required:
        if field in normalized:
            present += 1
            if not _is_empty_value(normalized[field]):
                non_empty += 1
    base = (present / max(1, len(required))) * 0.6 + (non_empty / max(1, len(required))) * 0.4
    penalty = min(0.5, 0.05 * len(errors))
    return round(max(0.0, min(1.0, base - penalty)), 4)


def validate_and_normalize_agent_output(
    agent_name: str,
    raw_output: str,
    contract: Optional[Dict[str, Any]] = None,
    strict_non_empty: bool = True,
) -> GovernanceResult:
    contract = contract or get_output_contract(agent_name)
    required = contract.get("required", {}) or {}
    aliases = contract.get("aliases", {}) or {}
    non_empty = set(contract.get("non_empty", []) or [])
    errors: List[str] = []
    repair_actions: List[str] = []

    parsed, warnings = parse_json_like_output(raw_output)
    data = _apply_aliases(parsed, aliases, repair_actions)
    normalized: Dict[str, Any] = {}

    for field, expected in required.items():
        originally_present = field in data
        if not originally_present:
            errors.append(f"missing_required_field:{field}")
            data[field] = None
            repair_actions.append(f"insert_missing_field:{field}")
        normalized[field] = _normalize_field(field, expected, data.get(field), repair_actions)
        if strict_non_empty and field in non_empty and _is_empty_value(normalized[field]):
            errors.append(f"empty_required_field:{field}")

    extras = {k: v for k, v in data.items() if k not in required}
    if extras:
        normalized["_extra"] = extras

    schema_valid = len(errors) == 0
    repaired = bool(repair_actions or warnings)
    return GovernanceResult(
        agent_name=agent_name,
        contract_name=agent_name,
        raw_output=raw_output,
        parsed_output=parsed,
        normalized_output=normalized,
        schema_valid=schema_valid,
        repaired=repaired,
        errors=errors,
        warnings=warnings,
        repair_actions=repair_actions,
        schema_quality_score=_schema_quality(required, normalized, errors),
    )


def normalized_output_text(result: GovernanceResult) -> str:
    return json.dumps(result.normalized_output, indent=2, ensure_ascii=False)


def build_schema_repair_prompt(raw_output: str, contract_name: str, contract: Optional[Dict[str, Any]] = None) -> str:
    """Generic repair prompt used only when schema validation fails."""
    instruction = schema_instruction_for_contract(contract_name, contract)
    return (
        "Repair the following model output so it satisfies the output contract exactly. "
        "Preserve the original meaning. Do not add unsupported facts. "
        "If a field is required but evidence is missing, use a short evidence-missing placeholder only when the field is non-empty; otherwise use [].\n\n"
        f"{instruction}\n\n"
        "MODEL_OUTPUT_TO_REPAIR:\n"
        f"{raw_output}"
    )
