from __future__ import annotations

import json
import re
from dataclasses import dataclass
from .llm_client import GroqClient, LLMResponse
from .optimizer import AggressiveSafeOptimizer, OptimizationResult, EvidenceGuardedOptimizer
from .evidence import EvidenceLedger, build_evidence_ledger, protected_terms_from_ledger


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
    """One ProdSync workflow agent.

    Important benchmark rule:
    - We do NOT inject the full evidence ledger into the output after the LLM call.
    - A previous version did that, producing fake 100% retention.
    - This version only normalizes/parses the LLM response and records schema validity.
    """

    def __init__(self, name: str, task: str):
        self.name = name
        self.task = task
        self._term_extractor = EvidenceGuardedOptimizer(level="guarded")

    def system_prompt(self) -> str:
        return (
            f"You are {self.name} inside ProdSync, a multi-agent hiring intelligence system. "
            "Return ONLY one compact JSON object. The first character must be { and the last character must be }. "
            "No markdown. No prose. No code fence. No comments. No trailing text. "
            "Be concise and deterministic. Do not invent experience. "
            "Use exact evidence terms from the evidence ledger when they are supported by context. "
            "Do not paraphrase ledger evidence. Preserve governance and risk terms exactly: human labels, acceptance metrics, model drift, concept drift, online monitoring, Kubernetes, MLOps, model registry, feature store."
        )

    def _ledger(self, context: str) -> EvidenceLedger:
        return build_evidence_ledger(context + "\n" + self.task)

    @staticmethod
    def _clip(items: list[str], limit: int) -> str:
        return ", ".join(items[:limit]) if items else "N/A"

    def user_prompt(self, context: str) -> str:
        ledger = self._ledger(context)
        # Keep prompt compact so 260 max output tokens can still close valid JSON.
        return (
            f"AGENT={self.name}\n"
            f"TASK={self.task}\n"
            f"MANDATORY_TERMS={self._clip(protected_terms_from_ledger(ledger, 40), 40)}\n"
            f"LEDGER_ENTITIES={self._clip(ledger.entities, 35)}\n"
            f"LEDGER_CONCERNS={self._clip(ledger.concerns, 18)}\n"
            f"LEDGER_REASONS={self._clip(ledger.reasons, 14)}\n"
            "RULES:\n"
            "1. JSON only. Start with { and end with }. Use double quotes only. No markdown.\n"
            "2. Keys exactly: final_score, shortlist_decision, evidence_terms, reasons, concerns, interview_plan. Do not add extra keys.\n"
            "3. shortlist_decision: Shortlist, Reject, Hold, or N/A.\n"
            "4. Keep arrays compact: evidence_terms<=24, reasons<=8, concerns<=10, interview_plan<=4.\n"
            "5. Copy supported MANDATORY_TERMS exactly; do not rename, merge, or paraphrase them.\n"
            "6. Negative/governance evidence is mandatory when supported: human labels, acceptance metrics, model drift, concept drift, online monitoring, Kubernetes, MLOps, model registry, feature store.\n"
            "7. If unsure, use N/A/null, not fabrication.\n"
            "SCHEMA_EXAMPLE={\"final_score\":7.0,\"shortlist_decision\":\"Shortlist\",\"evidence_terms\":[\"python\"],\"reasons\":[\"python fastapi sql skills\"],\"concerns\":[\"weak evidence for online monitoring\"],\"interview_plan\":[\"Ask about production monitoring\"]}\n"
            "CONTEXT:\n"
            f"{context}"
        )

    def _normalize_response(self, response: LLMResponse, context: str) -> LLMResponse:
        """Parse/normalize JSON without faking evidence retention.

        This function now tracks repair severity instead of one vague repaired count:
        - none: valid JSON and already follows required schema.
        - minor: valid JSON, only compact serialization/casing/list cleanup happened.
        - major: valid JSON but required schema keys/aliases/types had to be fixed.
        - fallback: model did not return parseable JSON; raw text was wrapped for honest reporting.

        No evidence terms are injected after the LLM call.
        """
        obj = _parse_json_object(response.text)
        schema_valid = obj is not None
        normalized = False
        major_repair = False
        minor_repair = False

        if obj is None:
            obj = {
                "final_score": _extract_score(response.text),
                "shortlist_decision": _normalize_decision(_extract_decision(response.text)) or "N/A",
                "evidence_terms": [],
                "reasons": [],
                "concerns": [],
                "interview_plan": [],
                "raw_output": response.text[:1200],
                "schema_valid": False,
            }
            normalized = True
            major_repair = True
        else:
            # Map compatible aliases if the model uses shorter names.
            alias_map = {
                "score": "final_score",
                "decision": "shortlist_decision",
                "evidence": "evidence_terms",
                "questions": "interview_plan",
                "risks": "concerns",
            }
            for old, new in alias_map.items():
                if old in obj and new not in obj:
                    obj[new] = obj.pop(old)
                    normalized = True
                    major_repair = True

            for key, default in {
                "final_score": None,
                "shortlist_decision": "N/A",
                "evidence_terms": [],
                "reasons": [],
                "concerns": [],
                "interview_plan": [],
            }.items():
                if key not in obj:
                    obj[key] = default
                    normalized = True
                    major_repair = True

            # Ensure array fields are arrays, but do not inject terms.
            for key in ["evidence_terms", "reasons", "concerns", "interview_plan"]:
                before = obj.get(key)
                if before is None:
                    obj[key] = []
                    normalized = True
                    major_repair = True
                elif not isinstance(before, list):
                    obj[key] = [str(before)]
                    normalized = True
                    major_repair = True
                else:
                    cleaned = [str(x).strip() for x in before if str(x).strip()]
                    if cleaned != before:
                        normalized = True
                        minor_repair = True
                    obj[key] = cleaned

            # Normalize decision casing.
            fixed_decision = _normalize_decision(str(obj.get("shortlist_decision", "N/A"))) or "N/A"
            if fixed_decision != obj.get("shortlist_decision"):
                obj["shortlist_decision"] = fixed_decision
                normalized = True
                minor_repair = True

            try:
                if obj.get("final_score") is not None:
                    fixed_score = float(obj.get("final_score"))
                    if fixed_score != obj.get("final_score"):
                        minor_repair = True
                    obj["final_score"] = fixed_score
            except Exception:
                obj["final_score"] = _extract_score(response.text)
                normalized = True
                major_repair = True

            obj["schema_valid"] = schema_valid

        normalized_text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        # Compact re-serialization alone is not a repair. It is normal output normalization.
        # Count only actual schema/key/type/casing changes as repair. This makes repair severity truthful.
        repair_flag = (not schema_valid) or normalized or major_repair or minor_repair
        if not schema_valid:
            repair_severity = "fallback"
        elif major_repair:
            repair_severity = "major"
        elif normalized or minor_repair:
            repair_severity = "minor"
        else:
            repair_severity = "none"

        return LLMResponse(
            text=normalized_text,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.total_tokens,
            real_api_call=response.real_api_call,
            repaired=repair_flag,
            raw_text=response.raw_text or response.text,
            json_mode_used=response.json_mode_used,
            fallback_used=response.fallback_used,
            schema_valid=schema_valid,
            repair_severity=repair_severity,
        )

    def run_baseline(self, llm: GroqClient, context: str, max_tokens: int) -> LLMResponse:
        resp = llm.generate(self.system_prompt(), self.user_prompt(context), max_tokens=max_tokens, temperature=0.0)
        return self._normalize_response(resp, context)

    def run_optimized(self, llm: GroqClient, context: str, optimizer: AggressiveSafeOptimizer, max_tokens: int) -> tuple[LLMResponse, OptimizationResult]:
        opt = optimizer.optimize(context, task=self.task)
        resp = llm.generate(self.system_prompt(), self.user_prompt(opt.optimized_text), max_tokens=max_tokens, temperature=0.0)
        return self._normalize_response(resp, opt.optimized_text), opt


def _parse_json_object(text: str) -> dict | None:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # Strip common code fences.
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _extract_score(text: str) -> float | None:
    patterns = [
        r"final[_ ]?score\s*[\":=]+\s*([0-9]+(?:\.[0-9]+)?)",
        r"score\s*[\":=]+\s*([0-9]+(?:\.[0-9]+)?)",
        r"([0-9]+(?:\.[0-9]+)?)\s*/\s*10",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.I)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
    return None


def _extract_decision(text: str) -> str | None:
    patterns = [
        r"shortlist_decision\s*[\":=]+\s*['\"]?([A-Za-z _-]+)",
        r"decision\s*[\":=]+\s*['\"]?([A-Za-z _-]+)",
        r"\b(Shortlist|Reject|Hold)\b",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.I)
        if m:
            return m.group(1).strip().split("\n")[0].strip(' ,.-_"\'')
    return None


def _normalize_decision(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip().lower()
    v = re.sub(r"[^a-z ]", "", v).strip()
    if v in {"shortlist", "yes", "selected", "recommend", "recommended", "hire", "proceed", "shortlisted"}:
        return "Shortlist"
    if v in {"reject", "no", "rejected", "do not shortlist", "not shortlist"}:
        return "Reject"
    if v in {"hold", "maybe", "manual review", "review", "borderline"}:
        return "Hold"
    if "shortlist" in v or "recommend" in v:
        return "Shortlist"
    if "reject" in v or "do not" in v:
        return "Reject"
    if "hold" in v or "review" in v:
        return "Hold"
    return None


def default_agents() -> list[ProdSyncAgent]:
    return [
        ProdSyncAgent("Resume Intelligence Agent", "Extract candidate strengths, weaknesses, skills, experience, project quality, and resume risks."),
        ProdSyncAgent("JD Matching Agent", "Compare candidate evidence against ML Engineer job requirements. Identify match score, missing skills, and hard blockers."),
        ProdSyncAgent("Interview Intelligence Agent", "Evaluate candidate interview answers for technical depth, communication, honesty, and readiness."),
        ProdSyncAgent("Recruiter Decision Agent", "Make final shortlist decision using all previous agent evidence. Return final_score, shortlist_decision, evidence_terms, reasons, concerns, interview_plan."),
    ]
