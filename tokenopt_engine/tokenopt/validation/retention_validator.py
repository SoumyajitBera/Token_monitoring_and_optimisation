import re
from typing import Dict, List, Tuple

from ..extractors import (
    extract_codes,
    extract_constraint_atoms,
    extract_entities,
    extract_keywords,
    extract_numbers,
)
from ..math_utils import clamp, cosine_text, coverage, weighted_average


def _norm_num(x: str) -> str:
    return re.sub(r"[^0-9.]", "", x or "")


def numeric_coverage(required: List[str], candidate_text: str) -> float:
    if not required:
        return 1.0
    cand_raw = candidate_text or ""
    cand_norm = re.sub(r"[^0-9.]", "", cand_raw)
    found = 0
    for item in required:
        if item in cand_raw:
            found += 1
            continue
        n = _norm_num(item)
        if n and n in cand_norm:
            found += 1
    return found / len(required)


def fuzzy_phrase_coverage(required: List[str], candidate_text: str, min_sim: float = 0.65) -> float:
    if not required:
        return 1.0
    found = 0
    cand = candidate_text.lower()
    for item in required:
        low = item.lower()
        if low in cand:
            found += 1
        elif cosine_text(item, candidate_text) >= min_sim:
            found += 1
    return found / len(required)


def _constraint_atom_coverage(atoms: List[Dict[str, object]], optimized: str) -> Tuple[float, Dict[str, object]]:
    """Score logical constraint preservation by semantic groups, not exact words.

    v1.2 punished safe rewrites like "must have X" -> "X is required".
    This scorer groups constraints into logical types and checks whether the
    optimized text preserves both the type and the nearby target terms.
    """
    if not atoms:
        return 1.0, {"atoms_checked": 0, "atoms_matched": 0, "matches": []}

    opt = (optimized or "").lower()
    matches = []
    matched_weight = 0.0
    total_weight = 0.0

    group_synonyms = {
        "negation": ["not", "no", "never", "without", "cannot", "can't", "must not", "should not", "do not", "exclude", "avoid"],
        "mandatory": ["must", "required", "mandatory", "need", "needs", "shall", "have to", "has to", "non negotiable", "essential"],
        "conditional": ["if", "only if", "unless", "provided", "when", "where", "condition", "subject to"],
        "threshold_min": ["at least", "minimum", "min", ">=", "greater than", "above", "over", "more than"],
        "threshold_max": ["at most", "maximum", "max", "<=", "less than", "below", "under", "no more than"],
        "priority": ["priority", "prioritize", "critical", "important", "focus", "primary"],
        "format": ["json", "schema", "structured", "return", "output", "format", "strict"],
    }

    for atom in atoms:
        group = str(atom.get("group", "generic"))
        terms = [str(t).lower() for t in atom.get("terms", []) if str(t).strip()]
        numbers = [str(n).lower() for n in atom.get("numbers", []) if str(n).strip()]
        weight = float(atom.get("weight", 1.0) or 1.0)
        total_weight += weight

        synonyms = group_synonyms.get(group, [])
        group_hit = any(s in opt for s in synonyms)

        # Target hit is intentionally lenient: constraints may be rewritten but
        # must keep nearby skill/code/number intent. If no target terms exist,
        # group preservation alone can count.
        term_hits = 0
        for term in terms[:8]:
            if len(term) < 3:
                continue
            if term in opt or cosine_text(term, optimized) >= 0.42:
                term_hits += 1
        num_hits = sum(1 for n in numbers if n in opt or _norm_num(n) in re.sub(r"[^0-9.]", "", opt))

        target_count = len([t for t in terms[:8] if len(t) >= 3]) + len(numbers)
        target_score = 1.0 if target_count == 0 else (term_hits + num_hits) / max(1, target_count)

        # A constraint is preserved if the logical group survives and at least
        # some target intent survives. For final decision/scoring, negation and
        # thresholds are weighted more strictly via atom weight.
        if group_hit and target_score >= 0.25:
            score = 1.0
        elif target_score >= 0.65:
            score = 0.85
        elif group_hit:
            score = 0.55
        else:
            score = max(0.0, min(0.45, target_score))

        matched_weight += weight * score
        matches.append({
            "group": group,
            "terms": terms[:8],
            "numbers": numbers,
            "group_hit": group_hit,
            "target_score": round(target_score, 4),
            "score": round(score, 4),
            "weight": weight,
        })

    return matched_weight / max(1e-9, total_weight), {
        "atoms_checked": len(atoms),
        "atoms_matched_soft": round(matched_weight, 4),
        "total_weight": round(total_weight, 4),
        "matches": matches[:25],
    }


class RetentionValidator:
    def score(self, original: str, optimized: str) -> Dict[str, object]:
        entities = list(set(extract_entities(original) + extract_codes(original)))
        numbers = extract_numbers(original)
        constraint_atoms = extract_constraint_atoms(original)
        keywords = extract_keywords(original)

        entity_retention = fuzzy_phrase_coverage(entities, optimized, min_sim=0.50)
        numeric_retention = numeric_coverage(numbers, optimized)
        constraint_retention, constraint_details = _constraint_atom_coverage(constraint_atoms, optimized)
        semantic_similarity = cosine_text(original, optimized)
        keyword_coverage = coverage(keywords, optimized)

        # Retention is deliberately safety-heavy, but not exact-word brittle.
        retention_score = weighted_average([
            (entity_retention, 0.22),
            (numeric_retention, 0.20),
            (constraint_retention, 0.23),
            (semantic_similarity, 0.20),
            (keyword_coverage, 0.15),
        ])

        return {
            "retention_score": clamp(retention_score),
            "entity_retention": clamp(entity_retention),
            "numeric_retention": clamp(numeric_retention),
            "constraint_retention": clamp(constraint_retention),
            "semantic_similarity": clamp(semantic_similarity),
            "keyword_coverage": clamp(keyword_coverage),
            "entities_checked": len(entities),
            "numbers_checked": len(numbers),
            "constraints_checked": len(constraint_atoms),
            "keywords_checked": len(keywords),
            "constraint_details": constraint_details,
        }
